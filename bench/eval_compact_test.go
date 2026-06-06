//go:build eval

// Redesign comparison: replays the recorded baseline trajectories under prefix
// caching for three arms — uncompressed, the original per-turn rewrite
// (compress.GuidelineCompressor), and cache-safe checkpoint compaction
// (compress.CompactingCompressor) — and reports net cost under caching for each, to
// show that checkpoint compaction removes the rewrite arm's cache loss. Deterministic,
// no API key, no spend:
//
//	COGNI_EVAL=1 go test -tags eval ./bench/ -run Compact -v
//
// Knobs: COGNI_HISTORY_BUDGETS, COGNI_SUMMARY_WORDS, COMPRESS_GUIDELINE,
// COGNI_AGENT_MODEL, COMPRESS_MODEL, COGNI_TARGET_FRACTION.
package bench

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/islamborghini/cogni2/internal/agent"
	"github.com/islamborghini/cogni2/internal/compress"
	"github.com/islamborghini/cogni2/internal/meter"
)

func TestCompactCacheComparison(t *testing.T) {
	if os.Getenv("COGNI_EVAL") != "1" {
		t.Skip("set COGNI_EVAL=1 to run the redesign cache comparison")
	}
	dir := filepath.Join("..", "bench", "runs", envOrDefault("COGNI_RUNS_SUBDIR", "stage3"), "baseline")
	trajs := loadBaselineTrajectories(dir)
	if len(trajs) == 0 {
		t.Skipf("no baseline trajectories in %s — run the end-to-end eval first", dir)
	}

	tok, err := meter.Default()
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	guideline, err := compress.LoadGuideline(os.Getenv("COMPRESS_GUIDELINE"))
	if err != nil {
		t.Fatalf("guideline: %v", err)
	}
	summaryWords := envInt("COGNI_SUMMARY_WORDS", 30)
	budgets := parseBudgets(envOrDefault("COGNI_HISTORY_BUDGETS", "2000,1500,1000,500"))
	targetFrac := 0.5
	if v, err := strconv.ParseFloat(os.Getenv("COGNI_TARGET_FRACTION"), 64); err == nil && v > 0 && v < 1 {
		targetFrac = v
	}

	agentP := PriceFor(envOrDefault("COGNI_AGENT_MODEL", "openai/gpt-oss-120b"))
	compP := PriceFor(envOrDefault("COMPRESS_MODEL", "llama-3.1-8b-instant"))
	if compP.InPer1M == 0 {
		compP = agentP
	}

	perTurnSystem := func(tr stage3Trajectory) int {
		if tr.Turns <= 0 {
			return 0
		}
		return tr.Buckets[meter.BucketSystem] / tr.Turns
	}

	// Uncompressed baseline cache cost per trajectory, under each caching assumption.
	type costs struct{ none, groq, frontier float64 }
	baseCost := map[string]costs{}
	var baseHitSum float64
	for _, tr := range trajs {
		cc, err := ReplayCacheCost(context.Background(), tr.Messages, perTurnSystem(tr), nil, 0, tok)
		if err != nil {
			t.Fatalf("baseline %s: %v", tr.TaskID, err)
		}
		baseCost[tr.TaskID] = costs{
			none:     CacheNetCostUSD(cc, agentP, 1.0, 0, compP),
			groq:     CacheNetCostUSD(cc, agentP, 0.5, 0, compP),
			frontier: CacheNetCostUSD(cc, agentP, 0.1, 0, compP),
		}
		baseHitSum += cc.HitRate()
	}
	baseHit := baseHitSum / float64(len(trajs))

	// newComp builds a fresh compressor of the named arm, with its own metering
	// summarizer so overhead is counted per (arm, budget).
	newComp := func(arm string) (compress.Compressor, *agent.MeteringSummarizer) {
		ms := &agent.MeteringSummarizer{Inner: SizeSummarizer{MaxWords: summaryWords}, Tok: tok}
		switch arm {
		case "rewrite":
			return &compress.GuidelineCompressor{Summarizer: ms, Tok: tok, Guideline: guideline}, ms
		default:
			return &compress.CompactingCompressor{Summarizer: ms, Tok: tok, Guideline: guideline, TargetFraction: targetFrac}, ms
		}
	}

	fmt.Printf("\n=== Stage 3 redesign under prompt caching (%d trajectories) ===\n", len(trajs))
	fmt.Printf("baseline cache hit rate = %.0f%% (append-only); checkpoint low-water = %.2f*budget\n", baseHit*100, targetFrac)
	fmt.Printf("%-8s %-11s %9s %9s %9s %12s %8s\n", "budget", "arm", "hit", "none%", "groq%", "frontier%", "engaged")

	var rows []CompactRow
	for _, budget := range budgets {
		for _, arm := range []string{"rewrite", "checkpoint"} {
			var noneRed, groqRed, frontRed, hitSum float64
			engaged := 0
			for _, tr := range trajs {
				comp, ms := newComp(arm)
				cc, err := ReplayCacheCost(context.Background(), tr.Messages, perTurnSystem(tr), comp, budget, tok)
				if err != nil {
					t.Fatalf("%s %s @ %d: %v", arm, tr.TaskID, budget, err)
				}
				ov := ms.InputTokens + ms.OutputTokens
				hitSum += cc.HitRate()
				if cc.Engaged {
					engaged++
				}
				bc := baseCost[tr.TaskID]
				noneRed += pctReduction(bc.none, CacheNetCostUSD(cc, agentP, 1.0, ov, compP))
				groqRed += pctReduction(bc.groq, CacheNetCostUSD(cc, agentP, 0.5, ov, compP))
				frontRed += pctReduction(bc.frontier, CacheNetCostUSD(cc, agentP, 0.1, ov, compP))
			}
			n := float64(len(trajs))
			row := CompactRow{
				Budget: budget, Arm: arm, TreatHitRate: hitSum / n,
				NoCacheCostRed: noneRed / n, GroqCostRed: groqRed / n, FrontierCostRed: frontRed / n,
				Engaged: engaged, Tasks: len(trajs),
			}
			fmt.Printf("%-8d %-11s %8.0f%% %8.1f%% %8.1f%% %11.1f%% %5d/%d\n",
				budget, arm, row.TreatHitRate*100, row.NoCacheCostRed, row.GroqCostRed, row.FrontierCostRed, engaged, len(trajs))
			rows = append(rows, row)
		}
	}

	// Controlled long-horizon model: the crossover the short real tasks cannot show.
	perTurnSyntheticSystem := 0
	for _, tr := range trajs {
		perTurnSyntheticSystem += perTurnSystem(tr)
	}
	perTurnSyntheticSystem /= len(trajs)
	horizon, err := HorizonSweep(tok, agentP, compP, perTurnSyntheticSystem, budgets[0], 300, summaryWords, []int{4, 8, 16, 32, 64})
	if err != nil {
		t.Fatalf("horizon sweep: %v", err)
	}
	fmt.Printf("\n--- synthetic horizon sweep (frontier 0.1, budget %d) ---\n", budgets[0])
	fmt.Printf("%-8s %10s %12s %14s\n", "rounds", "rw hit", "ck hit", "rw→ck cost")
	for _, h := range horizon {
		fmt.Printf("%-8d %9.0f%% %11.0f%% %+6.1f%% %+6.1f%%\n",
			h.Rounds, h.RewriteHitRate*100, h.CheckHitRate*100, h.RewriteFrontier, h.CheckFrontier)
	}

	set, err := Load(envOrDefault("COGNI_TASKS", "tasks.yaml"))
	if err != nil {
		t.Fatalf("load tasks: %v", err)
	}
	run := fmt.Sprintf("- replayed %d recorded baseline trajectories under prefix caching\n"+
		"- arms: uncompressed baseline, rewrite (GuidelineCompressor), checkpoint (CompactingCompressor, low-water %.2f*budget)\n"+
		"- agent model `%s`, summarizer `%s`; cache modeled at message granularity\n"+
		"- history budgets swept: %v",
		len(trajs), targetFrac, envOrDefault("COGNI_AGENT_MODEL", "openai/gpt-oss-120b"),
		envOrDefault("COMPRESS_MODEL", "llama-3.1-8b-instant"), budgets)
	md := RenderCompactReport(set, baseHit, rows, horizon, run)
	if err := os.MkdirAll("results", 0o755); err != nil {
		t.Fatalf("mkdir results: %v", err)
	}
	outName := envOrDefault("COGNI_RUNS_SUBDIR", "stage3") + "-cache-redesign.md"
	if err := os.WriteFile(filepath.Join("results", outName), []byte(md), 0o644); err != nil {
		t.Fatalf("write %s: %v", outName, err)
	}
}
