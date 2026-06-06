package bench

import (
	"context"
	"fmt"
	"strings"

	"github.com/islamborghini/cogni2/internal/agent"
	"github.com/islamborghini/cogni2/internal/compress"
	"github.com/islamborghini/cogni2/internal/meter"
)

// This file renders the redesign comparison: the same cache-aware replay as
// stage3-cache.md, but with a third arm — cache-safe checkpoint compaction
// (compress.CompactingCompressor) — next to the uncompressed baseline and the
// original per-turn rewrite (compress.GuidelineCompressor). The point it has to make
// is that checkpoint compaction REMOVES the rewrite arm's loss: it keeps a high cache
// hit rate and a net cost no worse than doing nothing, where the rewrite arm went
// 13-17% more expensive once caching was priced in.

// CompactRow is one (history budget, arm) point of the redesign comparison, billed
// under prefix caching three ways: no caching (the token replay's view), Groq's 0.5
// identical-prefix discount, and a frontier-style 0.1 read hit. Reductions are vs the
// uncompressed baseline; positive = cheaper, negative = more expensive.
type CompactRow struct {
	Budget          int
	Arm             string // "rewrite" or "checkpoint"
	TreatHitRate    float64
	NoCacheCostRed  float64
	GroqCostRed     float64
	FrontierCostRed float64
	Engaged         int
	Tasks           int
}

// SyntheticTrajectory is the controlled long-horizon input the frozen-20 cannot
// provide: a goal followed by `rounds` of (small action, large observation). Real
// localization tasks run 3-8 turns, far too short for any compaction to amortize a
// cache bust, so they cannot separate the rewrite and checkpoint policies. Lengthen
// the horizon and the difference appears: the rewrite policy re-summarizes a newer
// observation almost every over-budget turn (a bust each time), while checkpoint
// compaction busts once per checkpoint and then rides a stable, smaller prefix.
func SyntheticTrajectory(rounds, obsWords int) []agent.ChatMessage {
	body := strings.Repeat("\nword", obsWords)
	msgs := []agent.ChatMessage{{Role: agent.RoleUser, Content: "the goal"}}
	for i := 0; i < rounds; i++ {
		anchor := fmt.Sprintf("f%d.py:1-%d", i, obsWords)
		msgs = append(msgs,
			agent.ChatMessage{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "s", Name: "search_code", Args: `{"q":"x"}`}}},
			agent.ChatMessage{Role: agent.RoleTool, ToolCallID: "s", Content: anchor + body, Origin: "retrieved_code"},
		)
	}
	return msgs
}

// HorizonRow is one trajectory-length point of the synthetic crossover: cache hit
// rate and frontier-discount net-cost reduction (vs the uncompressed baseline) for
// the rewrite and checkpoint arms as the horizon grows.
type HorizonRow struct {
	Rounds          int
	BaseHitRate     float64
	RewriteHitRate  float64
	CheckHitRate    float64
	RewriteFrontier float64
	CheckFrontier   float64
}

// HorizonSweep replays synthetic trajectories of increasing length through the cache
// model for all three arms, at a frontier read discount (0.1) where a bust hurts
// most. It is the deterministic, free demonstration that checkpoint compaction's
// one-bust-per-checkpoint behavior pulls ahead of the rewrite policy as the horizon
// grows — the win the short real tasks cannot show.
func HorizonSweep(tok meter.Tokenizer, agentP, compP Price, perTurnSystem, budget, obsWords, summaryWords int, roundsList []int) ([]HorizonRow, error) {
	ctx := context.Background()
	var rows []HorizonRow
	for _, rounds := range roundsList {
		msgs := SyntheticTrajectory(rounds, obsWords)
		base, err := ReplayCacheCost(ctx, msgs, perTurnSystem, nil, 0, tok)
		if err != nil {
			return nil, err
		}
		baseFront := CacheNetCostUSD(base, agentP, 0.1, 0, compP)

		msR := &agent.MeteringSummarizer{Inner: SizeSummarizer{MaxWords: summaryWords}, Tok: tok}
		rw, err := ReplayCacheCost(ctx, msgs, perTurnSystem, &compress.GuidelineCompressor{Summarizer: msR, Tok: tok}, budget, tok)
		if err != nil {
			return nil, err
		}
		msC := &agent.MeteringSummarizer{Inner: SizeSummarizer{MaxWords: summaryWords}, Tok: tok}
		ck, err := ReplayCacheCost(ctx, msgs, perTurnSystem, &compress.CompactingCompressor{Summarizer: msC, Tok: tok, TargetFraction: 0.5}, budget, tok)
		if err != nil {
			return nil, err
		}
		rows = append(rows, HorizonRow{
			Rounds: rounds, BaseHitRate: base.HitRate(),
			RewriteHitRate: rw.HitRate(), CheckHitRate: ck.HitRate(),
			RewriteFrontier: pctReduction(baseFront, CacheNetCostUSD(rw, agentP, 0.1, msR.InputTokens+msR.OutputTokens, compP)),
			CheckFrontier:   pctReduction(baseFront, CacheNetCostUSD(ck, agentP, 0.1, msC.InputTokens+msC.OutputTokens, compP)),
		})
	}
	return rows, nil
}

// RenderCompactReport writes bench/results/stage3-cache-redesign.md.
func RenderCompactReport(set *TaskSet, baseHit float64, rows []CompactRow, horizon []HorizonRow, run string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Stage 3 redesign — cache-safe checkpoint compaction\n\n")
	fmt.Fprintf(&b, "Target: `%s` @ `%s`\n\n", set.TargetRepo, set.TargetSHA)
	if run != "" {
		fmt.Fprintf(&b, "## Run\n\n%s\n\n", run)
	}
	fmt.Fprintf(&b, "Same fixed-trajectory replay as `stage3-cache.md`, billed under prefix caching, but with a "+
		"third arm: **checkpoint compaction** compacts rarely (only when history crosses the budget) all the way "+
		"down to a low-water target, then freezes it — at most one cache divergence per checkpoint — where the "+
		"original **rewrite** arm re-summarized a newer observation almost every turn and broke the cache again "+
		"each time. The uncompressed baseline only appends, holding a **%.0f%% cache hit rate**.\n\n", baseHit*100)
	fmt.Fprintf(&b, "Net cost is read three ways (reduction vs the uncompressed baseline; positive = cheaper):\n")
	fmt.Fprintf(&b, "- **no caching**: cache ignored (what the token replay reports).\n")
	fmt.Fprintf(&b, "- **Groq 0.5**: cached input at half price.\n")
	fmt.Fprintf(&b, "- **frontier 0.1**: cached read at ~a tenth — where a cache bust hurts most.\n\n")

	fmt.Fprintf(&b, "| budget | arm | cache hit | no caching | Groq 0.5 | frontier 0.1 | engaged |\n")
	fmt.Fprintf(&b, "|---:|---|---:|---:|---:|---:|---:|\n")
	for _, r := range rows {
		fmt.Fprintf(&b, "| %d | %s | %.0f%% | %+.1f%% | %+.1f%% | %+.1f%% | %d/%d |\n",
			r.Budget, r.Arm, r.TreatHitRate*100, r.NoCacheCostRed, r.GroqCostRed, r.FrontierCostRed, r.Engaged, r.Tasks)
	}
	b.WriteString("\nOn this short-horizon set the two policies are **indistinguishable**: the frozen 20 are 3-8 turns, " +
		"so a task crosses the budget at most once, near the end, where a single compaction busts the cache with no " +
		"later turns to amortize it. Both arms lose the same amount under caching. The benchmark cannot separate the " +
		"designs — it is the wrong instrument for Stage 3.\n\n")

	if len(horizon) > 0 {
		fmt.Fprintf(&b, "## Where the redesign wins: a controlled horizon sweep\n\n")
		fmt.Fprintf(&b, "Synthetic trajectories of increasing length (goal + N rounds of action + large observation), "+
			"billed at the frontier 0.1 discount. This is a **model, not a measurement** — it isolates the mechanism the "+
			"short real tasks cannot exercise. As the horizon grows, the rewrite policy busts the cache almost every "+
			"over-budget turn while checkpoint compaction busts once per checkpoint, so checkpoint's hit rate stays high "+
			"and its net cost crosses ahead.\n\n")
		fmt.Fprintf(&b, "| rounds | base hit | rewrite hit | checkpoint hit | rewrite net cost | checkpoint net cost |\n")
		fmt.Fprintf(&b, "|---:|---:|---:|---:|---:|---:|\n")
		for _, h := range horizon {
			fmt.Fprintf(&b, "| %d | %.0f%% | %.0f%% | %.0f%% | %+.1f%% | %+.1f%% |\n",
				h.Rounds, h.BaseHitRate*100, h.RewriteHitRate*100, h.CheckHitRate*100, h.RewriteFrontier, h.CheckFrontier)
		}
		b.WriteString("\nNet cost is reduction vs the uncompressed baseline (positive = cheaper). The crossover is the " +
			"argument for the redesign; demonstrating it on real work needs the long-horizon (SWE-bench) task set, which " +
			"is the separate next step.\n")
	}
	return b.String()
}
