//go:build eval

// This is the Stage 2 end-to-end eval: it reuses the Stage 1 cached index (chunk
// budget unchanged, so no re-embedding), retrieves the frozen task set, and sweeps
// the skeleton-first assembly budget to produce bench/results/stage2.md. Retrieval
// is untouched, so recall@10 is identical to Stage 1 — reported only as a guard.
//
//	export COGNI_EVAL=1
//	export COGNI_BENCH_REPO=/path/to/django   # checked out at target_sha
//	export VOYAGE_API_KEY=…                    # or EMBED_PROVIDER=ollama …
//	go test -tags eval ./bench/ -run Skeleton -v -timeout 30m
package bench

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/islamborghini/cogni2/internal/meter"
	"github.com/islamborghini/cogni2/internal/parse"
	"github.com/islamborghini/cogni2/internal/retrieve"
	"github.com/islamborghini/cogni2/internal/skeleton"
)

// stage2Budgets is the assembly-budget sweep. The low end (2000) is there to find
// where chunks_dropped first crosses 0 — the eviction boundary is the finding. No
// single budget is published; that is deferred to Stage 3 (end-to-end success).
var stage2Budgets = []int{6000, 5000, 4000, 3000, 2000}

// canonicalBudget is the budget whose per-task run is persisted to bench/runs/stage2
// (parity with Stage 1); the rest of the curve is computed in-memory.
const canonicalBudget = 6000

func TestSkeleton(t *testing.T) {
	env := setupEvalIndex(t)
	ctx := context.Background()

	// Retrieve once per task — retrieval is identical across budgets — and reuse it
	// for the whole sweep. Recall is scored on the retrieved set, so it equals Stage
	// 1: a guard that the retriever still works, not a Stage 2 result.
	type retrievedTask struct {
		task      Task
		retrieved []retrieve.RetrievedChunk
	}
	var perTask []retrievedTask
	var locSum float64
	var locN int
	for _, task := range env.set.Tasks {
		retrieved, err := env.store.Retrieve(ctx, task.Query, retrievalK)
		if err != nil {
			t.Fatalf("retrieve %q: %v", task.ID, err)
		}
		perTask = append(perTask, retrievedTask{task: task, retrieved: retrieved})
		if task.Bucket == Localization {
			locSum += Recall(task.Gold, retrieved)
			locN++
		}
	}
	recall := 0.0
	if locN > 0 {
		recall = locSum / float64(locN)
	}

	// Stage 1 per-task tokens for the (b) total-reduction column.
	stage1Tok, err := loadStage1Tokens("..")
	if err != nil {
		t.Logf("stage 1 runs not found (%v); the (b) total-reduction column will read 0", err)
	}

	// Safety: parse-validity of every skeleton we actually emit (computed once,
	// budget-independent, at the canonical budget pass).
	skeletonsParsed, skeletonFailures := 0, 0

	var rows []BudgetRow
	for _, budget := range stage2Budgets {
		var assembledSum, droppedSum, tasksWithDrops int
		var skelRedSum, totalRedSum float64
		var skelRedN, totalRedN int

		for _, pt := range perTask {
			asm, err := skeleton.Assemble(pt.retrieved, budget, env.tok)
			if err != nil {
				t.Fatalf("assemble %q @ %d: %v", pt.task.ID, budget, err)
			}
			assembled := env.tok.Count(asm.Text)
			assembledSum += assembled
			droppedSum += asm.Dropped
			if asm.Dropped > 0 {
				tasksWithDrops++
			}

			// (a) skeletonization reduction: vs the same kept chunks as full bodies.
			if asm.KeptFullTokens > 0 {
				skelRedSum += float64(asm.KeptFullTokens-assembled) / float64(asm.KeptFullTokens) * 100
				skelRedN++
			}
			// (b) total reduction vs Stage 1 (when the run is available).
			if s1, ok := stage1Tok[pt.task.ID]; ok && s1 > 0 {
				totalRedSum += float64(s1-assembled) / float64(s1) * 100
				totalRedN++
			}

			if budget == canonicalBudget {
				m := meter.New(env.tok, 2, pt.task.ID)
				m.Add(meter.BucketRetrievedCode, asm.Text)
				if _, err := m.Persist(".."); err != nil {
					t.Fatalf("persist meter %q: %v", pt.task.ID, err)
				}
				for _, c := range pt.retrieved {
					s, err := skeleton.Skeleton(c)
					if err != nil {
						t.Fatalf("skeleton %q: %v", pt.task.ID, err)
					}
					if s == c.Content {
						continue // class_header or verbatim passthrough — unchanged
					}
					skeletonsParsed++
					if !skeletonParses(t, s) {
						skeletonFailures++
						t.Errorf("skeleton is not valid Python (task %q):\n%s", pt.task.ID, s)
					}
				}
			}
		}

		n := float64(len(perTask))
		row := BudgetRow{
			Budget:         budget,
			MeanTokens:     float64(assembledSum) / n,
			ChunksDropped:  droppedSum,
			TasksWithDrops: tasksWithDrops,
		}
		if skelRedN > 0 {
			row.SkelReductionPct = skelRedSum / float64(skelRedN)
		}
		if totalRedN > 0 {
			row.TotalReductionPct = totalRedSum / float64(totalRedN)
		}
		rows = append(rows, row)
	}

	res := Stage2Result{
		Recall:                recall,
		SkeletonsParsed:       skeletonsParsed,
		SkeletonParseFailures: skeletonFailures,
		Rows:                  rows,
	}

	if err := os.MkdirAll("results", 0o755); err != nil {
		t.Fatalf("mkdir results: %v", err)
	}
	run := fmt.Sprintf("- embedder: `%s` / `%s`\n- chunk budget: %d tokens (tiktoken), merge on\n"+
		"- corpus: %d files, %d chunks\n- k: %d\n- full-body fraction: %.2f (fixed)\n"+
		"- assembly budgets swept: %v; canonical run persisted at %d.",
		env.provider, env.model, env.maxTok, env.files, env.chunks, retrievalK,
		skeleton.FullBodyBudgetFraction, stage2Budgets, canonicalBudget)
	md := RenderMarkdownStage2(env.set, res, retrievalK, run)
	if err := os.WriteFile(filepath.Join("results", "stage2.md"), []byte(md), 0o644); err != nil {
		t.Fatalf("write stage2.md: %v", err)
	}

	fmt.Printf("\n=== Stage 2 (tokens down, retrieval and syntax intact) ===\n")
	fmt.Printf("recall@%d (localization, guard) = %.3f; skeletons valid = %d/%d\n",
		retrievalK, recall, skeletonsParsed-skeletonFailures, skeletonsParsed)
	for _, r := range rows {
		fmt.Printf("budget %d: mean_tokens=%.0f  (a) skel=%.1f%%  (b) total=%.1f%%  dropped=%d (%d tasks)\n",
			r.Budget, r.MeanTokens, r.SkelReductionPct, r.TotalReductionPct, r.ChunksDropped, r.TasksWithDrops)
	}
}

// skeletonParses reports whether a skeleton is valid Python (no error nodes) — the
// Stage 2 safety property.
func skeletonParses(t *testing.T, src string) bool {
	t.Helper()
	tree, err := parse.PythonTree([]byte(src))
	if err != nil {
		t.Fatalf("parse skeleton: %v", err)
	}
	defer tree.Close()
	return !tree.RootNode().HasError()
}
