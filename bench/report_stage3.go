package bench

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/islamborghini/cogni2/internal/meter"
)

// BucketCompression is the extra meter bucket Stage 3 introduces for the cost of
// the summarizer calls themselves. It is charged against the treatment arm so the
// reported reduction is net, not gross. It is NOT one of meter's four canonical
// buckets — it uses the meter's documented extra-bucket path.
const BucketCompression = "compression"

// Price is a model's list price in USD per 1M tokens.
type Price struct{ InPer1M, OutPer1M float64 }

// groqListPrices are list prices for the default Groq models (USD / 1M tokens),
// used only to turn measured usage into an illustrative net-cost figure. On the
// free tier the real bill is ~$0; the load-bearing guard is the cache-hit-rate
// delta between arms, not the dollar number. Edit if your plan's prices differ.
var groqListPrices = map[string]Price{
	"llama-3.3-70b-versatile": {0.59, 0.79},
	"llama-3.1-8b-instant":    {0.05, 0.08},
	"openai/gpt-oss-20b":      {0.075, 0.30},
	"openai/gpt-oss-120b":     {0.15, 0.60},
}

// PriceFor returns list prices for a model, or zero (cost reads 0) if unknown.
func PriceFor(model string) Price {
	if p, ok := groqListPrices[model]; ok {
		return p
	}
	return Price{}
}

// NetCostUSD applies Groq's 50% identical-prefix cache discount: cached input
// tokens bill at half price, the rest at full.
func NetCostUSD(prompt, cached, completion int, p Price) float64 {
	billedIn := float64(prompt-cached) + float64(cached)*0.5
	return billedIn*p.InPer1M/1e6 + float64(completion)*p.OutPer1M/1e6
}

// RunRecord is one (task, repeat) result under one arm — the unit the gated eval
// feeds the aggregator. Buckets is the per-bucket token sum across the run's
// calls (our tiktoken meter); the *Tokens fields are the API's real billed sums.
type RunRecord struct {
	TaskID           string
	Localization     bool
	Success          bool
	Buckets          map[string]int
	PromptTokens     int
	CachedTokens     int
	CompletionTokens int
	Summarized       int
	Dropped          int
}

// ArmResult aggregates one arm (baseline, or a treatment at a history budget)
// over all task-repeats.
type ArmResult struct {
	Label          string
	HistoryBudget  int // 0 for baseline
	Runs           int
	SuccessRate    float64 // localization macro-avg
	Buckets        map[string]float64
	MeanTotal      float64 // tiktoken-metered tokens per run
	MeanNetCostUSD float64 // cache-aware billed cost per run
	CacheHitRate   float64 // cached / prompt, pooled
	MeanSummarized float64
	MeanDropped    float64
}

// AggregateArm reduces per-run records into one arm result. model prices the
// billed usage. SuccessRate is over localization runs only (parity with Stage 1).
func AggregateArm(label string, budget int, model string, recs []RunRecord) ArmResult {
	ar := ArmResult{Label: label, HistoryBudget: budget, Runs: len(recs), Buckets: map[string]float64{}}
	if len(recs) == 0 {
		return ar
	}
	p := PriceFor(model)
	sumBuckets := map[string]int{}
	var sumTotal, sumPrompt, sumCached, sumCompletion, sumSum, sumDrop int
	var sumCost float64
	var locN, locWins int
	for _, r := range recs {
		if r.Localization {
			locN++
			if r.Success {
				locWins++
			}
		}
		for b, v := range r.Buckets {
			sumBuckets[b] += v
			sumTotal += v
		}
		sumPrompt += r.PromptTokens
		sumCached += r.CachedTokens
		sumCompletion += r.CompletionTokens
		sumSum += r.Summarized
		sumDrop += r.Dropped
		sumCost += NetCostUSD(r.PromptTokens, r.CachedTokens, r.CompletionTokens, p)
	}
	n := float64(len(recs))
	for b, v := range sumBuckets {
		ar.Buckets[b] = float64(v) / n
	}
	ar.MeanTotal = float64(sumTotal) / n
	ar.MeanNetCostUSD = sumCost / n
	ar.MeanSummarized = float64(sumSum) / n
	ar.MeanDropped = float64(sumDrop) / n
	if sumPrompt > 0 {
		ar.CacheHitRate = float64(sumCached) / float64(sumPrompt)
	}
	if locN > 0 {
		ar.SuccessRate = float64(locWins) / float64(locN)
	}
	return ar
}

// Stage3Result is everything the report needs.
type Stage3Result struct {
	Baseline            ArmResult
	Treatments          []ArmResult
	Stage1RetrievedCode float64 // mean tokens from runs/stage1 (0 if not run)
	Stage2RetrievedCode float64 // mean tokens from runs/stage2 (0 if not run)
	AgentModel          string
	CompressModel       string
	Tolerance           float64 // success-rate tolerance for a valid treatment
}

// pctReduction is how much treat is below base, as a percentage of base.
func pctReduction(base, treat float64) float64 {
	if base <= 0 {
		return 0
	}
	return (base - treat) / base * 100
}

// pctChange is the signed change from base to val (positive = larger/worse cost).
func pctChange(base, val float64) float64 {
	if base <= 0 {
		return 0
	}
	return (val - base) / base * 100
}

// BestValidTreatment returns the treatment that holds success within tolerance of
// the baseline AND cuts the most total tokens, or false if none holds.
func (r Stage3Result) BestValidTreatment() (ArmResult, bool) {
	var best ArmResult
	found := false
	for _, t := range r.Treatments {
		if t.SuccessRate < r.Baseline.SuccessRate-r.Tolerance {
			continue
		}
		if !found || pctReduction(r.Baseline.MeanTotal, t.MeanTotal) > pctReduction(r.Baseline.MeanTotal, best.MeanTotal) {
			best, found = t, true
		}
	}
	return best, found
}

// reportBuckets is the bucket order the Stage 3 tables use.
var reportBuckets = []string{
	meter.BucketRetrievedCode, meter.BucketHistory, meter.BucketSystem, meter.BucketOutput, BucketCompression,
}

// RenderMarkdownStage3 produces bench/results/stage3.md: the run config, the
// Pareto headline (success held + tokens down + net-cost guard), the three-stage
// continuity table, the full per-arm budget sweep, and the honesty notes.
func RenderMarkdownStage3(set *TaskSet, res Stage3Result, run string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Stage 3 — + ACON-style history/observation compression\n\n")
	fmt.Fprintf(&b, "Target: `%s` @ `%s`\n\n", set.TargetRepo, set.TargetSHA)
	if run != "" {
		fmt.Fprintf(&b, "## Run\n\n%s\n\n", run)
	}
	sortTreatmentsByBudget(res.Treatments) // loosest budget first

	base := res.Baseline
	fmt.Fprintf(&b, "## Headline: same success rate, fewer tokens (end-to-end)\n\n")
	fmt.Fprintf(&b, "Stage 3 is the first stage measured end-to-end, so it is the first that can earn the "+
		"\"same success rate, N%% fewer tokens\" claim — scoped to this short-horizon localization set on an "+
		"open model (see notes).\n\n")

	if best, ok := res.BestValidTreatment(); ok {
		held := best.SuccessRate >= base.SuccessRate-res.Tolerance
		fmt.Fprintf(&b, "- **success_rate**: baseline **%.3f** → treatment (budget %d) **%.3f** %s\n",
			base.SuccessRate, best.HistoryBudget, best.SuccessRate, checkMark(held))
		fmt.Fprintf(&b, "- **mean_total_tokens**: baseline **%.0f** → treatment **%.0f**\n", base.MeanTotal, best.MeanTotal)
		fmt.Fprintf(&b, "- **pct_total_reduction (raw, tiktoken)**: **%.1f%%** at held success\n",
			pctReduction(base.MeanTotal, best.MeanTotal))
		fmt.Fprintf(&b, "- **net billed cost guard**: baseline $%.4f → treatment $%.4f (**%+.1f%%**); "+
			"cache-hit-rate %.1f%% → %.1f%%. ", base.MeanNetCostUSD, best.MeanNetCostUSD,
			pctChange(base.MeanNetCostUSD, best.MeanNetCostUSD), base.CacheHitRate*100, best.CacheHitRate*100)
		b.WriteString("Rewriting history busts the prefix cache, so a raw-token win can shrink (or reverse) once " +
			"caching is priced in — this guard is the honest bottom line.\n\n")
	} else {
		fmt.Fprintf(&b, "- No swept budget held success within ±%.3f of the baseline (**%.3f**). "+
			"The run is **invalid** as a Pareto claim — tune the guideline/budget, do not relax the test.\n\n",
			res.Tolerance, base.SuccessRate)
	}

	// Three-stage continuity (retrieved_code is the through-line from Stages 1–2).
	fmt.Fprintf(&b, "## Tokens per bucket (Stage 1 → 2 → 3)\n\n")
	fmt.Fprintf(&b, "Stages 1–2 were offline single-shot (only `retrieved_code` measured). Stage 3 sums every "+
		"bucket across all turns of the loop, so the columns are not directly comparable — the through-line is "+
		"`retrieved_code`, and the Stage 3 headline is the **total**.\n\n")
	fmt.Fprintf(&b, "| bucket | Stage 1 | Stage 2 | Stage 3 baseline | Stage 3 treatment |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|---:|\n")
	best, ok := res.BestValidTreatment()
	for _, bk := range reportBuckets {
		s1, s2 := "—", "—"
		if bk == meter.BucketRetrievedCode {
			s1 = fmt.Sprintf("%.0f", res.Stage1RetrievedCode)
			s2 = fmt.Sprintf("%.0f", res.Stage2RetrievedCode)
		}
		treat := "—"
		if ok {
			treat = fmt.Sprintf("%.0f", best.Buckets[bk])
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %.0f | %s |\n", bk, s1, s2, base.Buckets[bk], treat)
	}
	fmt.Fprintf(&b, "| **total** | — | — | **%.0f** | %s |\n", base.MeanTotal, totalCell(ok, best.MeanTotal))
	fmt.Fprintf(&b, "| **success_rate** | n/a (recall guard) | n/a (parse-valid) | **%.3f** | %s |\n\n",
		base.SuccessRate, successCell(ok, best.SuccessRate))

	// Full budget sweep.
	fmt.Fprintf(&b, "## Budget sweep (baseline vs each history budget)\n\n")
	fmt.Fprintf(&b, "Success within ±%.3f of baseline = a valid Pareto point. Below that, the token savings are "+
		"not free and the row is flagged.\n\n", res.Tolerance)
	fmt.Fprintf(&b, "| arm | success | mean_total | raw reduction | net cost | cache hit | summarized | dropped | valid |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|---:|---:|---:|---:|:--:|\n")
	fmt.Fprintf(&b, "| baseline | %.3f | %.0f | — | $%.4f | %.1f%% | — | — | — |\n",
		base.SuccessRate, base.MeanTotal, base.MeanNetCostUSD, base.CacheHitRate*100)
	for _, t := range res.Treatments {
		valid := t.SuccessRate >= base.SuccessRate-res.Tolerance
		fmt.Fprintf(&b, "| budget %d | %.3f | %.0f | %.1f%% | $%.4f | %.1f%% | %.1f | %.1f | %s |\n",
			t.HistoryBudget, t.SuccessRate, t.MeanTotal, pctReduction(base.MeanTotal, t.MeanTotal),
			t.MeanNetCostUSD, t.CacheHitRate*100, t.MeanSummarized, t.MeanDropped, checkMark(valid))
	}
	b.WriteString("\n")

	fmt.Fprintf(&b, "## Notes\n\n")
	fmt.Fprintf(&b, "- **Short-horizon caveat**: the frozen 20 are localization tasks (~3–8 turns); accumulated "+
		"history is modest, so expect a smaller reduction than ACON's long-horizon 26–54%%. The swappable loader "+
		"makes a SWE-bench Lite subset the next step for a stronger claim.\n")
	fmt.Fprintf(&b, "- **Open-model caveat**: success is measured on `%s`, not a frontier model — absolute "+
		"success may differ and frontier generalization is unproven (ACON found open/small models gain *more* from "+
		"compression, so this is a favorable setting, not a dodge).\n", res.AgentModel)
	fmt.Fprintf(&b, "- **`compression` bucket** is the summarizer (`%s`) overhead, charged into the treatment "+
		"total so the reduction is net.\n", res.CompressModel)
	fmt.Fprintf(&b, "- **Net cost** uses list prices and Groq's 50%% prefix-cache discount; on the free tier the "+
		"dollar figure is ~$0, so read the **cache-hit-rate delta** as the real signal.\n")
	return b.String()
}

func checkMark(ok bool) string {
	if ok {
		return "✓"
	}
	return "✗"
}

func totalCell(ok bool, v float64) string {
	if !ok {
		return "—"
	}
	return fmt.Sprintf("**%.0f**", v)
}

func successCell(ok bool, v float64) string {
	if !ok {
		return "—"
	}
	return fmt.Sprintf("**%.3f**", v)
}

// meanRetrievedCode reads the mean retrieved_code tokens persisted by an earlier
// stage's eval (bench/runs/stage<N>/*.json under root), for the continuity table.
// Returns 0 if that stage has not been run.
func meanRetrievedCode(root string, stage int) float64 {
	dir := filepath.Join(root, "bench", "runs", fmt.Sprintf("stage%d", stage))
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	sum, n := 0, 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		var rec meter.Record
		if json.Unmarshal(data, &rec) != nil {
			continue
		}
		sum += rec.Buckets[meter.BucketRetrievedCode]
		n++
	}
	if n == 0 {
		return 0
	}
	return float64(sum) / float64(n)
}

// sortTreatmentsByBudget orders treatments loosest-budget-first for the report.
func sortTreatmentsByBudget(ts []ArmResult) {
	sort.SliceStable(ts, func(i, j int) bool { return ts[i].HistoryBudget > ts[j].HistoryBudget })
}
