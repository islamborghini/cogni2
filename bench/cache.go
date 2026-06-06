package bench

import (
	"context"
	"fmt"
	"hash/fnv"
	"strings"

	"github.com/islamborghini/cogni2/internal/agent"
	"github.com/islamborghini/cogni2/internal/compress"
	"github.com/islamborghini/cogni2/internal/meter"
)

// This file adds the cache-aware view of Stage 3. The token replay (replay.go)
// asks "how many tokens does the loop re-send?"; this asks the question that
// actually sets the bill on a caching frontier API: "how many of those tokens are
// served from the prefix cache, and how many does compression force the model to
// recompute?" Prefix caching keys on an exact, growing prefix; the uncompressed
// baseline only ever appends, so it keeps a near-total hit rate, while rewriting an
// older observation moves the first-difference earlier and re-bills the whole tail
// after it. That is the trap a history-rewriting Stage 3 has to survive — measured,
// like everything else here, by replaying a fixed trajectory.

// promptBlock is one cacheable unit of a request: the system+tools preamble, or one
// accumulated message, as sent on a given model turn. Two consecutive turns share
// cache up to the first block whose Hash differs; Tok is the block's token count.
type promptBlock struct {
	Tok  int
	Hash uint64
}

// hashBlock fingerprints a block's exact bytes; the separator keeps concatenations
// of different field splits from colliding.
func hashBlock(parts ...string) uint64 {
	h := fnv.New64a()
	for _, p := range parts {
		_, _ = h.Write([]byte(p))
		_, _ = h.Write([]byte{0})
	}
	return h.Sum64()
}

// msgBlock renders one message as the cache sees it: role, content, and any tool
// call (name + arguments). When compression rewrites an observation, its Content
// changes, so its Hash changes — exactly where the prefix cache breaks.
func msgBlock(m agent.ChatMessage, tok meter.Tokenizer) promptBlock {
	parts := []string{m.Role, m.Content}
	for _, tc := range m.ToolCalls {
		parts = append(parts, tc.Name, tc.Args)
	}
	return promptBlock{Tok: msgTokens(m, tok), Hash: hashBlock(parts...)}
}

// replayPrompts walks a recorded trajectory exactly as the live loop would — and,
// with comp != nil, compresses the accumulated history after each turn, identically
// to ReplayCost — returning the ordered input-block sequence the model is sent on
// each model turn, the output tokens generated, and whether compression engaged.
// Both ReplayCost (token totals) and ReplayCacheCost (cache-aware billing) are pure
// functions of these per-turn prompts, so the two views can never drift apart.
func replayPrompts(ctx context.Context, messages []agent.ChatMessage, perTurnSystem int, comp compress.Compressor, budget int, tok meter.Tokenizer) (prompts [][]promptBlock, output int, engaged bool, err error) {
	// The system prompt + tool schemas are re-sent every turn and identical across
	// turns and arms, so they carry a constant hash: the stable cached head.
	sys := promptBlock{Tok: perTurnSystem, Hash: hashBlock("system+tools")}
	rebuild := make([]agent.ChatMessage, 0, len(messages))
	i := 0
	for i < len(messages) {
		m := messages[i]
		if m.Role != agent.RoleAssistant {
			rebuild = append(rebuild, m) // the goal (first user turn) and any stray
			i++
			continue
		}

		// A model turn: it is sent system + tools + everything accumulated so far.
		p := make([]promptBlock, 0, len(rebuild)+1)
		p = append(p, sys)
		for _, mm := range rebuild {
			p = append(p, msgBlock(mm, tok))
		}
		prompts = append(prompts, p)
		output += msgTokens(m, tok)

		rebuild = append(rebuild, m)
		i++
		for i < len(messages) && messages[i].Role == agent.RoleTool {
			rebuild = append(rebuild, messages[i])
			i++
		}

		if comp == nil {
			continue
		}
		turns := make([]compress.Turn, len(rebuild))
		for j, mm := range rebuild {
			turns[j] = compress.Turn{Role: mm.Role, Content: mm.Content, Step: j, Kind: mm.Origin, Compressed: mm.Compressed}
		}
		res, e := comp.Compress(ctx, turns, budget)
		if e != nil {
			return nil, 0, false, e
		}
		for j := range res.History {
			rebuild[j].Content = res.History[j].Content
			rebuild[j].Compressed = res.History[j].Compressed
		}
		if res.Summarized > 0 || res.Dropped > 0 {
			engaged = true
		}
	}
	return prompts, output, engaged, nil
}

// CacheCost is the cache-aware billing of one replayed arm. Cached input tokens are
// served from the prefix cache (billed at a discount); Fresh input is recomputed at
// full price; Output is generated tokens, never cached.
type CacheCost struct {
	Cached  int
	Fresh   int
	Output  int
	Engaged bool
}

// HitRate is the share of input tokens served from the prefix cache.
func (c CacheCost) HitRate() float64 {
	in := c.Cached + c.Fresh
	if in == 0 {
		return 0
	}
	return float64(c.Cached) / float64(in)
}

// ReplayCacheCost replays a trajectory and bills input under prefix caching: each
// model turn reuses the cache up to the first block that differs from the previous
// turn's prompt, then pays full price from there on. Append-only growth (the
// uncompressed baseline) keeps a near-total hit rate; rewriting an older observation
// moves the first difference earlier, so the whole tail after the rewrite is
// re-billed. Modeled at message granularity against the immediately preceding turn,
// which is how an agent loop's monotonic prefix actually populates the cache.
func ReplayCacheCost(ctx context.Context, messages []agent.ChatMessage, perTurnSystem int, comp compress.Compressor, budget int, tok meter.Tokenizer) (CacheCost, error) {
	prompts, output, engaged, err := replayPrompts(ctx, messages, perTurnSystem, comp, budget, tok)
	if err != nil {
		return CacheCost{}, err
	}
	cc := CacheCost{Output: output, Engaged: engaged}
	var prev []promptBlock
	for _, p := range prompts {
		matched := true
		for k, b := range p {
			if matched && k < len(prev) && prev[k].Hash == b.Hash {
				cc.Cached += b.Tok
			} else {
				matched = false // first divergence: cache is broken for the rest
				cc.Fresh += b.Tok
			}
		}
		prev = p
	}
	return cc, nil
}

// CacheRow is one history-budget point of the cache experiment: how much the
// net-cost reduction Stage 3 claims survives once prefix caching is priced in, read
// three ways — no caching (the figure the token replay reports), Groq's 0.5
// identical-prefix discount, and a frontier-style 0.1 read hit. A reduction that is
// positive without caching but negative with it is the cache trap firing.
type CacheRow struct {
	Budget          int
	BaseHitRate     float64
	TreatHitRate    float64
	NoCacheCostRed  float64
	GroqCostRed     float64
	FrontierCostRed float64
	Engaged         int
	Tasks           int
}

// RenderCacheReport writes bench/results/stage3-cache.md: the same trajectory replay
// as Stage 3, but billed under prefix caching, so the headline is whether the cost
// win is real on a caching API or an artifact of ignoring the cache.
func RenderCacheReport(set *TaskSet, baseHit float64, rows []CacheRow, run string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Stage 3 under prompt caching — does the cost win survive?\n\n")
	fmt.Fprintf(&b, "Target: `%s` @ `%s`\n\n", set.TargetRepo, set.TargetSHA)
	if run != "" {
		fmt.Fprintf(&b, "## Run\n\n%s\n\n", run)
	}
	fmt.Fprintf(&b, "Same fixed-trajectory replay as the Stage 3 result, but input is billed under "+
		"**prefix caching**: each turn reuses the cache up to the first message that changed, and pays full "+
		"price after it. The uncompressed baseline only appends, so it keeps a **%.0f%% cache hit rate**; "+
		"compression rewrites an older observation, which moves the first change earlier and re-bills the tail. "+
		"The question is whether the net-cost reduction is real on a caching API.\n\n", baseHit*100)
	fmt.Fprintf(&b, "Net-cost reduction is read three ways:\n")
	fmt.Fprintf(&b, "- **no caching**: what the token replay reports (cache ignored).\n")
	fmt.Fprintf(&b, "- **Groq 0.5**: cached input billed at half price (the free-tier eval's own discount).\n")
	fmt.Fprintf(&b, "- **frontier 0.1**: cached read at ~a tenth — where the cache is worth most and a bust hurts most.\n\n")

	fmt.Fprintf(&b, "| history budget | cache hit (base→treat) | net cost: no caching | Groq 0.5 | frontier 0.1 | engaged |\n")
	fmt.Fprintf(&b, "|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range rows {
		fmt.Fprintf(&b, "| %d | %.0f%% → %.0f%% | %+.1f%% | %+.1f%% | %+.1f%% | %d/%d |\n",
			r.Budget, r.BaseHitRate*100, r.TreatHitRate*100,
			r.NoCacheCostRed, r.GroqCostRed, r.FrontierCostRed, r.Engaged, r.Tasks)
	}
	b.WriteString("\nPositive = cheaper than the uncompressed baseline; negative = more expensive. ")
	b.WriteString("A column that flips from positive (no caching) to negative (with caching) is the prompt-cache " +
		"trap: compression saves cheap, already-cached tokens and pays full price to recompute the tail it moved.\n")
	return b.String()
}

// CacheNetCostUSD prices a cache split: fresh input at full price, cached input
// scaled by cachedMul (Groq's identical-prefix discount is 0.5; a frontier read
// hit is nearer 0.1, which makes the cache worth more and a bust costlier). Output
// is full price. Pass overheadTok/overheadPrice to fold in the summarizer's own
// (uncached, cheap-model) tokens so the figure is net, not gross.
func CacheNetCostUSD(c CacheCost, p Price, cachedMul float64, overheadTok int, overheadPrice Price) float64 {
	billedIn := float64(c.Fresh) + float64(c.Cached)*cachedMul
	cost := billedIn*p.InPer1M/1e6 + float64(c.Output)*p.OutPer1M/1e6
	cost += float64(overheadTok) * overheadPrice.InPer1M / 1e6
	return cost
}
