//go:build eval

// Cache-aware Stage 3 eval: it replays the SAME recorded baseline trajectories as
// the token replay (bench/runs/stage3/baseline/*.json), but bills input under prefix
// caching — so it answers the question the token count cannot: once the cache is
// live, does history compression still lower net cost, or does rewriting the prefix
// hand the savings back? Deterministic, no API key, no spend:
//
//	COGNI_EVAL=1 go test -tags eval ./bench/ -run ReplayCache -v
//
// Knobs (shared with the token replay): COGNI_HISTORY_BUDGETS, COGNI_SUMMARY_WORDS,
// COMPRESS_GUIDELINE, COGNI_AGENT_MODEL, COMPRESS_MODEL.
package bench

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/islamborghini/cogni2/internal/agent"
	"github.com/islamborghini/cogni2/internal/compress"
	"github.com/islamborghini/cogni2/internal/meter"
)

func TestReplayCache(t *testing.T) {
	if os.Getenv("COGNI_EVAL") != "1" {
		t.Skip("set COGNI_EVAL=1 to run the cache-aware replay")
	}
	dir := filepath.Join("..", "bench", "runs", "stage3", "baseline")
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

	agentP := PriceFor(envOrDefault("COGNI_AGENT_MODEL", "openai/gpt-oss-120b"))
	compP := PriceFor(envOrDefault("COMPRESS_MODEL", "llama-3.1-8b-instant"))
	if compP.InPer1M == 0 {
		compP = agentP // unknown compressor: price like the agent (conservative)
	}

	perTurnSystem := func(tr stage3Trajectory) int {
		if tr.Turns <= 0 {
			return 0
		}
		return tr.Buckets[meter.BucketSystem] / tr.Turns
	}

	// Baseline cache cost per trajectory, under each caching assumption.
	type costs struct{ none, groq, frontier float64 }
	baseCost := map[string]costs{}
	var baseHitSum float64
	for _, tr := range trajs {
		cc, err := ReplayCacheCost(context.Background(), tr.Messages, perTurnSystem(tr), nil, 0, tok)
		if err != nil {
			t.Fatalf("baseline cache %s: %v", tr.TaskID, err)
		}
		baseCost[tr.TaskID] = costs{
			none:     CacheNetCostUSD(cc, agentP, 1.0, 0, compP),
			groq:     CacheNetCostUSD(cc, agentP, 0.5, 0, compP),
			frontier: CacheNetCostUSD(cc, agentP, 0.1, 0, compP),
		}
		baseHitSum += cc.HitRate()
	}
	baseHit := baseHitSum / float64(len(trajs))

	fmt.Printf("\n=== Stage 3 under prompt caching (%d trajectories) ===\n", len(trajs))
	fmt.Printf("baseline cache hit rate = %.0f%% (append-only)\n", baseHit*100)
	fmt.Printf("%-8s %14s %12s %10s %12s %9s\n", "budget", "treat_hit", "none%", "groq0.5%", "frontier0.1%", "engaged")

	var rows []CacheRow
	for _, budget := range budgets {
		var noneRed, groqRed, frontRed, treatHitSum float64
		engaged := 0
		for _, tr := range trajs {
			ms := &agent.MeteringSummarizer{Inner: SizeSummarizer{MaxWords: summaryWords}, Tok: tok}
			comp := &compress.GuidelineCompressor{Summarizer: ms, Tok: tok, Guideline: guideline}
			cc, err := ReplayCacheCost(context.Background(), tr.Messages, perTurnSystem(tr), comp, budget, tok)
			if err != nil {
				t.Fatalf("treatment cache %s @ %d: %v", tr.TaskID, budget, err)
			}
			ov := ms.InputTokens + ms.OutputTokens
			treatHitSum += cc.HitRate()
			if cc.Engaged {
				engaged++
			}
			bc := baseCost[tr.TaskID]
			tn := CacheNetCostUSD(cc, agentP, 1.0, ov, compP)
			tg := CacheNetCostUSD(cc, agentP, 0.5, ov, compP)
			tf := CacheNetCostUSD(cc, agentP, 0.1, ov, compP)
			noneRed += pctReduction(bc.none, tn)
			groqRed += pctReduction(bc.groq, tg)
			frontRed += pctReduction(bc.frontier, tf)
		}
		n := float64(len(trajs))
		row := CacheRow{
			Budget: budget, BaseHitRate: baseHit, TreatHitRate: treatHitSum / n,
			NoCacheCostRed: noneRed / n, GroqCostRed: groqRed / n, FrontierCostRed: frontRed / n,
			Engaged: engaged, Tasks: len(trajs),
		}
		fmt.Printf("%-8d %13.0f%% %11.1f%% %9.1f%% %11.1f%% %6d/%d\n",
			budget, row.TreatHitRate*100, row.NoCacheCostRed, row.GroqCostRed, row.FrontierCostRed, engaged, len(trajs))
		rows = append(rows, row)
	}

	set, err := Load("tasks.yaml")
	if err != nil {
		t.Fatalf("load tasks: %v", err)
	}
	run := fmt.Sprintf("- replayed %d recorded baseline trajectories under prefix caching\n"+
		"- agent model `%s`, summarizer `%s`; cache modeled at message granularity\n"+
		"- history budgets swept: %v",
		len(trajs), envOrDefault("COGNI_AGENT_MODEL", "openai/gpt-oss-120b"),
		envOrDefault("COMPRESS_MODEL", "llama-3.1-8b-instant"), budgets)
	md := RenderCacheReport(set, baseHit, rows, run)
	if err := os.MkdirAll("results", 0o755); err != nil {
		t.Fatalf("mkdir results: %v", err)
	}
	if err := os.WriteFile(filepath.Join("results", "stage3-cache.md"), []byte(md), 0o644); err != nil {
		t.Fatalf("write stage3-cache.md: %v", err)
	}
}
