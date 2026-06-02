package bench

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/islamborghini/cogni2/internal/meter"
)

// BudgetRow is one assembly-budget point on the Stage 2 curve.
type BudgetRow struct {
	Budget            int
	MeanTokens        float64
	SkelReductionPct  float64 // (a) vs the same kept chunks as full bodies — isolates skeletonization
	TotalReductionPct float64 // (b) vs Stage-1 per-task tokens — total effect at this budget
	ChunksDropped     int
	TasksWithDrops    int
}

// Stage2Result is everything the Stage 2 report needs.
type Stage2Result struct {
	// Recall is the localization macro-avg — a GUARD that retrieval is unchanged,
	// NOT a quality claim. It is identical to Stage 1 by construction.
	Recall                float64
	SkeletonsParsed       int
	SkeletonParseFailures int
	Rows                  []BudgetRow
}

// DropBoundaryBudget returns the highest swept budget at which any chunk was
// evicted (chunks_dropped > 0), or 0 if none. The eviction boundary is the
// headline finding of the sweep: at/below it, recall@k overstates what the agent
// actually sees.
func (r Stage2Result) DropBoundaryBudget() int {
	best := 0
	for _, row := range r.Rows {
		if row.ChunksDropped > 0 && row.Budget > best {
			best = row.Budget
		}
	}
	return best
}

// RenderMarkdownStage2 produces bench/results/stage2.md: the run config, the safety
// summary (recall guard + parse-validity + eviction boundary), and the budget-sweep
// curve with both reduction columns. run is a markdown snippet describing how the
// numbers were produced, so the report stands on its own.
func RenderMarkdownStage2(set *TaskSet, res Stage2Result, k int, run string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Stage 2 — cAST + skeleton-first compression\n\n")
	fmt.Fprintf(&b, "Target: `%s` @ `%s`\n\n", set.TargetRepo, set.TargetSHA)
	if run != "" {
		fmt.Fprintf(&b, "## Run\n\n%s\n\n", run)
	}

	fmt.Fprintf(&b, "## Result: tokens down, retrieval and syntax intact\n\n")
	fmt.Fprintf(&b, "Stage 2 changes only how retrieved chunks are *rendered*; retrieval is unchanged. "+
		"This is **not** a quality claim — \"no quality loss\" / \"same success rate\" stay unproven until "+
		"end-to-end (Stage 3).\n\n")
	fmt.Fprintf(&b, "- **retrieval intact**: recall@%d (localization, macro-avg) = **%.3f**, identical to Stage 1 "+
		"by construction — reported only as a guard that the retriever still works.\n", k, res.Recall)
	valid := res.SkeletonsParsed - res.SkeletonParseFailures
	fmt.Fprintf(&b, "- **syntax intact**: %d/%d emitted skeletons parse as valid Python (%d failures).\n",
		valid, res.SkeletonsParsed, res.SkeletonParseFailures)
	if boundary := res.DropBoundaryBudget(); boundary > 0 {
		fmt.Fprintf(&b, "- **eviction boundary**: chunks first dropped at budget **%d** — at/below it, recall@%d "+
			"overstates what the agent sees (see chunks_dropped).\n", boundary, k)
	} else {
		fmt.Fprintf(&b, "- **eviction boundary**: no chunk dropped at any swept budget — every retrieved chunk "+
			"is present at least as a skeleton + anchor.\n")
	}
	b.WriteString("\n")

	fmt.Fprintf(&b, "## Budget sweep\n\n")
	fmt.Fprintf(&b, "Lead column is **(a) skeletonization reduction** — vs the same kept chunks rendered as full "+
		"bodies at the same budget. It isolates skeleton-first compression and survives the \"isn't this just a "+
		"smaller budget\" challenge. **(b) total reduction @ budget** is vs the Stage-1 baseline and *includes* the "+
		"budget cap — the total effect at that budget, not skeletonization's effect alone.\n\n")
	fmt.Fprintf(&b, "| budget | mean_tokens | (a) skeletonization reduction | (b) total reduction @ budget | chunks_dropped | tasks_with_drops |\n")
	fmt.Fprintf(&b, "|---:|---:|---:|---:|---:|---:|\n")
	for _, row := range res.Rows {
		fmt.Fprintf(&b, "| %d | %.0f | %.1f%% | %.1f%% | %d | %d |\n",
			row.Budget, row.MeanTokens, row.SkelReductionPct, row.TotalReductionPct, row.ChunksDropped, row.TasksWithDrops)
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "Safety claim for Stage 2 = **parse-validity + chunks_dropped**, not recall@%d. "+
		"recall@%d only certifies that retrieval is unchanged.\n", k, k)
	return b.String()
}

// loadStage1Tokens reads the per-task retrieved_code token counts persisted by the
// Stage 1 eval (bench/runs/stage1/*.json under root), for the (b) total-reduction
// column. Returns an error if the directory is absent (Stage 1 not yet run).
func loadStage1Tokens(root string) (map[string]int, error) {
	dir := filepath.Join(root, "bench", "runs", "stage1")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	out := map[string]int{}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, err
		}
		var rec meter.Record
		if err := json.Unmarshal(data, &rec); err != nil {
			return nil, err
		}
		out[rec.TaskID] = rec.Buckets[meter.BucketRetrievedCode]
	}
	return out, nil
}
