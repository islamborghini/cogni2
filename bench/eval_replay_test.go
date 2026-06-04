//go:build eval

// Stage 3 trajectory-replay eval: it replays the recorded BASELINE trajectories
// (bench/runs/stage3/baseline/*.json) and measures the tokens the loop would pay
// to feed each one back, uncompressed vs compressed, across a history-budget
// sweep. Because the action sequence is fixed, success is held by construction —
// this isolates compression's token effect with none of the live-run noise.
//
// It needs no API key or index, only the recorded trajectories and a tokenizer,
// so the compressed observation size is modeled deterministically (SizeSummarizer):
//
//	COGNI_EVAL=1 go test -tags eval ./bench/ -run Replay -v
//
// Knobs: COGNI_HISTORY_BUDGETS, COGNI_SUMMARY_WORDS, COMPRESS_GUIDELINE.
package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/islamborghini/cogni2/internal/agent"
	"github.com/islamborghini/cogni2/internal/compress"
	"github.com/islamborghini/cogni2/internal/meter"
)

func TestReplay(t *testing.T) {
	if os.Getenv("COGNI_EVAL") != "1" {
		t.Skip("set COGNI_EVAL=1 to run the replay eval")
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

	perTurnSystem := func(tr stage3Trajectory) int {
		if tr.Turns <= 0 {
			return 0
		}
		return tr.Buckets[meter.BucketSystem] / tr.Turns
	}

	// Uncompressed baseline cost per trajectory.
	uncTotals := map[string]int{}
	var uncSum float64
	for _, tr := range trajs {
		r, err := ReplayCost(context.Background(), tr.Messages, perTurnSystem(tr), nil, 0, tok)
		if err != nil {
			t.Fatalf("replay %s: %v", tr.TaskID, err)
		}
		uncTotals[tr.TaskID] = r.Total()
		uncSum += float64(r.Total())
	}
	uncMean := uncSum / float64(len(trajs))

	// Prices to weight overhead (cheap compress model) vs agent tokens (expensive).
	agentInP := PriceFor(envOrDefault("COGNI_AGENT_MODEL", "openai/gpt-oss-120b")).InPer1M
	compInP := PriceFor(envOrDefault("COMPRESS_MODEL", "llama-3.1-8b-instant")).InPer1M
	if compInP == 0 {
		compInP = agentInP // unknown compressor: price it like the agent (conservative)
	}

	var rows []ReplayRow
	fmt.Printf("\n=== Stage 3 (trajectory replay, %d trajectories) ===\n", len(trajs))
	fmt.Printf("uncompressed mean total = %.0f tokens\n", uncMean)
	fmt.Printf("%-8s %10s %10s %10s %10s %9s\n", "budget", "net_tok%", "gross_tok%", "net_$%", "overhead", "engaged")
	for _, budget := range budgets {
		var totalSum, ovSum, netRedSum, grossRedSum, uncCost, cmpCost float64
		engaged := 0
		for _, tr := range trajs {
			ms := &agent.MeteringSummarizer{Inner: SizeSummarizer{MaxWords: summaryWords}, Tok: tok}
			comp := &compress.GuidelineCompressor{Summarizer: ms, Tok: tok, Guideline: guideline}
			r, err := ReplayCost(context.Background(), tr.Messages, perTurnSystem(tr), comp, budget, tok)
			if err != nil {
				t.Fatalf("replay %s @ %d: %v", tr.TaskID, budget, err)
			}
			r.Overhead = ms.InputTokens + ms.OutputTokens
			totalSum += float64(r.Total())
			ovSum += float64(r.Overhead)
			if r.Engaged {
				engaged++
			}
			u := uncTotals[tr.TaskID]
			if u <= 0 {
				continue
			}
			agentTok := r.Input + r.Output // expensive-model tokens (no overhead)
			netRedSum += float64(u-r.Total()) / float64(u) * 100
			grossRedSum += float64(u-agentTok) / float64(u) * 100
			uncCost += float64(u) * agentInP
			cmpCost += float64(agentTok)*agentInP + float64(r.Overhead)*compInP
		}
		n := float64(len(trajs))
		costRed := 0.0
		if uncCost > 0 {
			costRed = (uncCost - cmpCost) / uncCost * 100
		}
		fmt.Printf("%-8d %9.1f%% %9.1f%% %9.1f%% %10.0f %6d/%d\n",
			budget, netRedSum/n, grossRedSum/n, costRed, ovSum/n, engaged, len(trajs))
		rows = append(rows, ReplayRow{
			Budget: budget, MeanTotal: totalSum / n,
			NetTokenPct: netRedSum / n, GrossTokenPct: grossRedSum / n, CostRedPct: costRed,
			MeanOverhead: ovSum / n, TasksEngaged: engaged, Tasks: len(trajs),
		})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Budget > rows[j].Budget })

	set, err := Load("tasks.yaml")
	if err != nil {
		t.Fatalf("load tasks: %v", err)
	}
	run := fmt.Sprintf("- replayed %d recorded baseline trajectories\n"+
		"- compressed-observation size modeled as: anchor + first %d words (deterministic, no LLM)\n"+
		"- history budgets swept: %v",
		len(trajs), summaryWords, budgets)
	md := RenderReplayStage3(set, uncMean, rows, run)
	if err := os.MkdirAll("results", 0o755); err != nil {
		t.Fatalf("mkdir results: %v", err)
	}
	if err := os.WriteFile(filepath.Join("results", "stage3.md"), []byte(md), 0o644); err != nil {
		t.Fatalf("write stage3.md: %v", err)
	}
}

func loadBaselineTrajectories(dir string) []stage3Trajectory {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []stage3Trajectory
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		var tr stage3Trajectory
		if json.Unmarshal(data, &tr) == nil && len(tr.Messages) > 0 {
			out = append(out, tr)
		}
	}
	return out
}
