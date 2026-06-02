package bench

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/islamborghini/cogni2/internal/meter"
)

func TestNetCostUSD(t *testing.T) {
	// 1000 prompt of which 800 cached (billed at 50%), + 100 completion.
	// billedIn = 200 + 400 = 600 → 600*1/1e6 = 6e-4; out = 100*2/1e6 = 2e-4.
	got := NetCostUSD(1000, 800, 100, Price{InPer1M: 1.0, OutPer1M: 2.0})
	if math.Abs(got-0.0008) > 1e-9 {
		t.Errorf("NetCostUSD = %v, want 0.0008", got)
	}
	if NetCostUSD(1000, 0, 0, PriceFor("unknown-model")) != 0 {
		t.Error("unknown model should price to 0")
	}
}

func TestAggregateArm(t *testing.T) {
	recs := []RunRecord{
		{TaskID: "a", Localization: true, Success: true,
			Buckets: map[string]int{meter.BucketRetrievedCode: 100, meter.BucketHistory: 50}, PromptTokens: 200, CachedTokens: 100, CompletionTokens: 20},
		{TaskID: "b", Localization: true, Success: false,
			Buckets: map[string]int{meter.BucketRetrievedCode: 200, meter.BucketHistory: 100}, PromptTokens: 400, CachedTokens: 0, CompletionTokens: 40},
	}
	ar := AggregateArm("baseline", 0, "llama-3.1-8b-instant", recs)
	if ar.SuccessRate != 0.5 {
		t.Errorf("success rate = %v, want 0.5", ar.SuccessRate)
	}
	if ar.Buckets[meter.BucketRetrievedCode] != 150 || ar.MeanTotal != 225 {
		t.Errorf("means = %+v / total %v, want rc 150 total 225", ar.Buckets, ar.MeanTotal)
	}
	// pooled cache-hit = 100 / (200+400) = 16.7%
	if math.Abs(ar.CacheHitRate-100.0/600.0) > 1e-9 {
		t.Errorf("cache hit = %v, want %v", ar.CacheHitRate, 100.0/600.0)
	}
}

func TestBestValidTreatment(t *testing.T) {
	res := Stage3Result{
		Baseline:  ArmResult{Label: "baseline", SuccessRate: 0.80, MeanTotal: 1000},
		Tolerance: 0.05,
		Treatments: []ArmResult{
			{Label: "b4000", HistoryBudget: 4000, SuccessRate: 0.80, MeanTotal: 800}, // valid, 20% off
			{Label: "b3000", HistoryBudget: 3000, SuccessRate: 0.78, MeanTotal: 600}, // valid, 40% off — best
			{Label: "b2000", HistoryBudget: 2000, SuccessRate: 0.60, MeanTotal: 500}, // invalid (success dropped)
		},
	}
	best, ok := res.BestValidTreatment()
	if !ok || best.HistoryBudget != 3000 {
		t.Fatalf("best valid = %+v ok=%v, want budget 3000", best, ok)
	}

	none := Stage3Result{Baseline: ArmResult{SuccessRate: 0.9, MeanTotal: 1000}, Tolerance: 0.05,
		Treatments: []ArmResult{{HistoryBudget: 2000, SuccessRate: 0.5, MeanTotal: 400}}}
	if _, ok := none.BestValidTreatment(); ok {
		t.Error("no treatment should be valid when success collapses")
	}
}

func TestRenderMarkdownStage3(t *testing.T) {
	set := &TaskSet{TargetRepo: "github.com/django/django", TargetSHA: "abc123"}
	res := Stage3Result{
		Baseline: ArmResult{Label: "baseline", SuccessRate: 0.80, MeanTotal: 12000, MeanNetCostUSD: 0.01, CacheHitRate: 0.7,
			Buckets: map[string]float64{meter.BucketRetrievedCode: 8000, meter.BucketHistory: 3000, meter.BucketSystem: 700, meter.BucketOutput: 300}},
		Treatments: []ArmResult{
			{Label: "b3000", HistoryBudget: 3000, SuccessRate: 0.80, MeanTotal: 9000, MeanNetCostUSD: 0.011, CacheHitRate: 0.4,
				Buckets: map[string]float64{meter.BucketRetrievedCode: 5000, meter.BucketHistory: 3000, meter.BucketSystem: 700, meter.BucketOutput: 300, BucketCompression: 500}, MeanSummarized: 4, MeanDropped: 1},
		},
		Stage1RetrievedCode: 3554, Stage2RetrievedCode: 3350,
		AgentModel: "llama-3.3-70b-versatile", CompressModel: "llama-3.1-8b-instant", Tolerance: 0.05,
	}
	md := RenderMarkdownStage3(set, res, "- agent: `llama-3.3-70b-versatile`")
	for _, want := range []string{
		"Stage 3", "abc123", "success_rate", "0.800", "mean_total_tokens",
		"pct_total_reduction", "net billed cost guard", "cache-hit-rate",
		"3554", "3350", "budget 3000", "compression", "llama-3.3-70b-versatile",
		"Short-horizon caveat", "Open-model caveat",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("stage3 report missing %q\n---\n%s", want, md)
		}
	}
}

func TestMeanRetrievedCode(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "bench", "runs", "stage1")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for id, n := range map[string]int{"a": 100, "b": 300} {
		rec := meter.Record{TaskID: id, Stage: 1, Buckets: map[string]int{meter.BucketRetrievedCode: n}, Total: n}
		data, _ := json.MarshalIndent(rec, "", "  ")
		if err := os.WriteFile(filepath.Join(dir, id+".json"), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if got := meanRetrievedCode(root, 1); got != 200 {
		t.Errorf("meanRetrievedCode = %v, want 200", got)
	}
	if got := meanRetrievedCode(root, 2); got != 0 {
		t.Errorf("absent stage should be 0, got %v", got)
	}
}
