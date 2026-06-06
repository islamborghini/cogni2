package bench

import (
	"context"
	"fmt"
	"strings"

	"github.com/islamborghini/cogni2/internal/agent"
	"github.com/islamborghini/cogni2/internal/compress"
	"github.com/islamborghini/cogni2/internal/meter"
)

// Replay measures Stage 3 the way Stages 1–2 were measured: on FIXED inputs,
// changing only the rendering. It takes a recorded agent trajectory (the exact
// sequence of actions and observations) and computes the tokens the loop would
// pay to feed it back each turn — uncompressed vs compressed. Because the action
// sequence is held constant, the final answer (and thus success) is unchanged by
// construction, so the token delta is pure compression effect with none of the
// run-to-run nondeterminism that swamps a live A/B.

// ReplayResult is the token cost of replaying one trajectory.
type ReplayResult struct {
	Input    int  // history/context re-sent across turns (what compression affects)
	Output   int  // generated tokens (identical across arms)
	Overhead int  // the summarizer's own tokens (the compression bucket)
	Engaged  bool // did compression actually run at some turn
}

// Total is the net token cost: re-sent context + output + compression overhead.
func (r ReplayResult) Total() int { return r.Input + r.Output + r.Overhead }

// msgTokens counts a message's content plus its tool-call name/arguments.
func msgTokens(m agent.ChatMessage, tok meter.Tokenizer) int {
	n := tok.Count(m.Content)
	for _, tc := range m.ToolCalls {
		n += tok.Count(tc.Name) + tok.Count(tc.Args)
	}
	return n
}

// ReplayCost replays a recorded message log and sums the tokens the loop pays. At
// each model turn it charges perTurnSystem (the system prompt + tool schemas,
// re-sent every turn and identical across arms) plus the current context. With
// comp != nil it compresses the accumulated history to budget after each turn —
// incrementally, exactly as the live loop does. Overhead is filled in by the
// caller from its MeteringSummarizer.
func ReplayCost(ctx context.Context, messages []agent.ChatMessage, perTurnSystem int, comp compress.Compressor, budget int, tok meter.Tokenizer) (ReplayResult, error) {
	prompts, output, engaged, err := replayPrompts(ctx, messages, perTurnSystem, comp, budget, tok)
	if err != nil {
		return ReplayResult{}, err
	}
	r := ReplayResult{Output: output, Engaged: engaged}
	for _, p := range prompts {
		for _, b := range p {
			r.Input += b.Tok
		}
	}
	return r, nil
}

// SizeSummarizer models a compressed observation's token footprint without an LLM,
// so the replay is deterministic and free: keep the first line (the path:line
// anchor / key fact) plus the first MaxWords words of the body. A token
// measurement only needs the summary's SIZE; real-LLM summaries are a separate
// validation step. MaxWords <= 0 keeps only the first line.
type SizeSummarizer struct{ MaxWords int }

// Summarize implements compress.Summarizer.
func (s SizeSummarizer) Summarize(_ context.Context, text, _ string) (string, error) {
	head, body, _ := strings.Cut(text, "\n")
	words := strings.Fields(body)
	if s.MaxWords > 0 && len(words) > s.MaxWords {
		words = words[:s.MaxWords]
	}
	if len(words) == 0 {
		return head, nil
	}
	return head + " " + strings.Join(words, " "), nil
}

// ReplayRow is one budget point on the replay curve, in three honest views.
type ReplayRow struct {
	Budget        int
	MeanTotal     float64 // mean compressed total tokens (incl. overhead)
	NetTokenPct   float64 // raw-token reduction, overhead at face value — can be negative
	GrossTokenPct float64 // agent-context reduction, ignoring overhead
	CostRedPct    float64 // net cost reduction (overhead priced on the cheap compressor)
	MeanOverhead  float64
	TasksEngaged  int
	Tasks         int
}

// bestCostRow returns the row with the largest net cost reduction.
func bestCostRow(rows []ReplayRow) ReplayRow {
	var best ReplayRow
	for i, r := range rows {
		if i == 0 || r.CostRedPct > best.CostRedPct {
			best = r
		}
	}
	return best
}

// RenderReplayStage3 writes bench/results/stage3.md: the trajectory-replay result,
// framed (like Stage 2) as tokens-down with success held by construction, with the
// headline on NET COST.
func RenderReplayStage3(set *TaskSet, uncompressedMean float64, rows []ReplayRow, run string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Stage 3 — ACON-style history compression (trajectory replay)\n\n")
	fmt.Fprintf(&b, "Target: `%s` @ `%s`\n\n", set.TargetRepo, set.TargetSHA)
	if run != "" {
		fmt.Fprintf(&b, "## Run\n\n%s\n\n", run)
	}

	fmt.Fprintf(&b, "## Result: ~%.0f%% lower net cost, success held by construction\n\n", bestCostRow(rows).CostRedPct)
	fmt.Fprintf(&b, "Measured by **replaying fixed recorded agent trajectories** and changing only whether "+
		"accumulated history is compressed — the Stage 1/2 discipline (fixed inputs, vary the rendering). The "+
		"action sequence is identical in both arms, so the final answer and success are unchanged **by "+
		"construction**; the token delta is pure compression effect, free of the run-to-run nondeterminism that "+
		"dominates a live A/B. Whether compressed context would change the agent's *decisions* is the separate "+
		"end-to-end question (deferred).\n\n")
	fmt.Fprintf(&b, "Read three ways, because they say different things:\n")
	fmt.Fprintf(&b, "- **net cost** (headline): overhead priced on the cheap compressor model, agent tokens on the "+
		"agent model — the bill you actually pay.\n")
	fmt.Fprintf(&b, "- **gross tokens**: how much smaller the (expensive) agent context gets, ignoring overhead.\n")
	fmt.Fprintf(&b, "- **net tokens**: raw count with the summarizer's tokens charged at face value — near "+
		"break-even on this short-horizon set, because the one-time summarization overhead barely amortizes.\n\n")
	fmt.Fprintf(&b, "- **uncompressed mean total**: %.0f tokens/trajectory\n\n", uncompressedMean)

	fmt.Fprintf(&b, "## History-budget sweep\n\n")
	fmt.Fprintf(&b, "| history budget | net cost | gross tokens | net tokens | mean overhead | trajectories compressed |\n")
	fmt.Fprintf(&b, "|---:|---:|---:|---:|---:|---:|\n")
	for _, row := range rows {
		fmt.Fprintf(&b, "| %d | **%.1f%%** | %.1f%% | %.1f%% | %.0f | %d/%d |\n",
			row.Budget, row.CostRedPct, row.GrossTokenPct, row.NetTokenPct, row.MeanOverhead, row.TasksEngaged, row.Tasks)
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "Percentages are macro-averages of per-trajectory reductions vs. the uncompressed replay. "+
		"The win is in **cost**, not raw token volume: compression shrinks the expensive agent context ~%.0f%% "+
		"and offloads summarization to a model several times cheaper. Compressed observation size is modeled "+
		"deterministically (anchor + first words); real-LLM summary validation is a separate step.\n",
		bestCostRow(rows).GrossTokenPct)
	return b.String()
}
