//go:build eval

// This is the Stage 3 end-to-end eval: it runs the frozen task set through the
// real multi-turn agent loop (internal/agent) twice — an uncompressed BASELINE
// arm and a TREATMENT arm per swept history budget — and reports tokens at a
// fixed success rate. It reuses the Stage 1/2 cached index for retrieval and an
// OpenAI-compatible model (Groq by default) for the agent + compressor.
//
//	export COGNI_EVAL=1
//	export COGNI_BENCH_REPO=/path/to/django      # checked out at target_sha
//	export VOYAGE_API_KEY=…                       # query embedding (or EMBED_PROVIDER=ollama …)
//	export GROQ_API_KEY=…                         # or LLM_API_KEY (+ LLM_BASE_URL for another provider)
//	# n=1 smoke first (respects Groq free-tier limits), then widen:
//	COGNI_EVAL_N=1 COGNI_HISTORY_BUDGETS=3000 go test -tags eval ./bench/ -run EndToEnd -v -timeout 60m
//
// Knobs: COGNI_AGENT_MODEL (llama-3.3-70b-versatile), COMPRESS_MODEL
// (llama-3.1-8b-instant), COGNI_HISTORY_BUDGETS ("4000,3000,2000"), COGNI_EVAL_N,
// COGNI_MAX_TURNS, COGNI_ASSEMBLY_BUDGET, COGNI_SUCCESS_TOLERANCE, COMPRESS_GUIDELINE.
//
// Trajectories are cached under bench/runs/stage3/<arm>/<task>__r<rep>.json: a run
// that already has a file is reused, so the eval is resumable across days against
// the rate-limited free tier and re-grading/reporting costs nothing.
package bench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/islamborghini/cogni2/internal/agent"
	"github.com/islamborghini/cogni2/internal/compress"
	"github.com/islamborghini/cogni2/internal/retrieve"
)

func TestEndToEnd(t *testing.T) {
	env := setupEvalIndex(t)
	ctx := context.Background()
	repo := os.Getenv("COGNI_BENCH_REPO") // validated by setupEvalIndex

	agentModel := envOrDefault("COGNI_AGENT_MODEL", "llama-3.3-70b-versatile")
	compressModel := envOrDefault("COMPRESS_MODEL", "llama-3.1-8b-instant")
	budgets := parseBudgets(envOrDefault("COGNI_HISTORY_BUDGETS", "3000"))
	repeats := envInt("COGNI_EVAL_N", 1)
	maxTurns := envInt("COGNI_MAX_TURNS", agent.DefaultMaxTurns)
	assemblyBudget := envInt("COGNI_ASSEMBLY_BUDGET", canonicalBudget)
	tolerance := envFloat("COGNI_SUCCESS_TOLERANCE", 0.05)

	guideline, err := compress.LoadGuideline(os.Getenv("COMPRESS_GUIDELINE"))
	if err != nil {
		t.Fatalf("load guideline: %v", err)
	}
	chat, err := agent.FromEnv(agentModel)
	if err != nil {
		t.Fatalf("agent model: %v", err)
	}
	compChat, err := agent.FromEnv(compressModel)
	if err != nil {
		t.Fatalf("compress model: %v", err)
	}
	tools := []agent.Tool{
		agent.NewSearchCodeTool(env.store, retrievalK, assemblyBudget, env.tok),
		agent.NewReadFileTool(repo, 400),
		agent.NewSubmitAnswerTool(),
		agent.NewSubmitAliasTool("commentary"), // gpt-oss sometimes names its final call "commentary"
	}

	type arm struct {
		label  string
		budget int
	}
	arms := []arm{{"baseline", 0}}
	for _, b := range budgets {
		arms = append(arms, arm{fmt.Sprintf("budget-%d", b), b})
	}

	records := map[string][]RunRecord{}
	for _, a := range arms {
		for _, task := range env.set.Tasks {
			for rep := 0; rep < repeats; rep++ {
				rec := runOrLoad(t, ctx, runParams{
					arm: a.label, budget: a.budget, repeat: rep, task: task,
					chat: chat, compChat: compChat, tok: env.tok, tools: tools,
					guideline: guideline, maxTurns: maxTurns,
				})
				records[a.label] = append(records[a.label], rec)
			}
		}
	}

	base := AggregateArm("baseline", 0, agentModel, records["baseline"])
	var treatments []ArmResult
	for _, b := range budgets {
		label := fmt.Sprintf("budget-%d", b)
		treatments = append(treatments, AggregateArm(label, b, agentModel, records[label]))
	}
	res := Stage3Result{
		Baseline:            base,
		Treatments:          treatments,
		Stage1RetrievedCode: meanRetrievedCode("..", 1),
		Stage2RetrievedCode: meanRetrievedCode("..", 2),
		AgentModel:          agentModel,
		CompressModel:       compressModel,
		Tolerance:           tolerance,
	}

	if err := os.MkdirAll("results", 0o755); err != nil {
		t.Fatalf("mkdir results: %v", err)
	}
	run := fmt.Sprintf("- agent model: `%s`; compressor: `%s` (provider %s)\n"+
		"- retrieval: %s/%s, k=%d, assembly budget %d (Stage 1/2 reused)\n"+
		"- history budgets swept: %v; repeats: %d; max turns: %d; success tolerance: ±%.3f\n"+
		"- success = cited locations cover all gold (localization, recall == 1.0)",
		agentModel, compressModel, envOrDefault("LLM_BASE_URL", "groq"),
		env.provider, env.model, retrievalK, assemblyBudget, budgets, repeats, maxTurns, tolerance)
	md := RenderMarkdownStage3(env.set, res, run)
	if err := os.WriteFile(filepath.Join("results", "stage3.md"), []byte(md), 0o644); err != nil {
		t.Fatalf("write stage3.md: %v", err)
	}

	fmt.Printf("\n=== Stage 3 (same success rate, fewer tokens) ===\n")
	fmt.Printf("baseline: success=%.3f mean_total=%.0f net=$%.4f cache=%.1f%%\n",
		base.SuccessRate, base.MeanTotal, base.MeanNetCostUSD, base.CacheHitRate*100)
	for _, tr := range treatments {
		fmt.Printf("budget %d: success=%.3f mean_total=%.0f reduction=%.1f%% net=$%.4f cache=%.1f%% summarized=%.1f dropped=%.1f\n",
			tr.HistoryBudget, tr.SuccessRate, tr.MeanTotal, pctReduction(base.MeanTotal, tr.MeanTotal),
			tr.MeanNetCostUSD, tr.CacheHitRate*100, tr.MeanSummarized, tr.MeanDropped)
	}
}

// runParams bundles one (task, arm, repeat) run's inputs.
type runParams struct {
	arm       string
	budget    int
	repeat    int
	task      Task
	chat      agent.Model
	compChat  compress.Summarizer
	tok       interface{ Count(string) int }
	tools     []agent.Tool
	guideline string
	maxTurns  int
}

// stage3Trajectory is the persisted per-run record: enough to re-grade and report
// offline, plus the full transcript for the guideline-refinement script.
type stage3Trajectory struct {
	TaskID           string              `json:"task_id"`
	Arm              string              `json:"arm"`
	Budget           int                 `json:"budget"`
	Repeat           int                 `json:"repeat"`
	Query            string              `json:"query"`
	Localization     bool                `json:"localization"`
	Success          bool                `json:"success"`
	Recall           float64             `json:"recall"`
	StopReason       string              `json:"stop_reason"`
	Turns            int                 `json:"turns"`
	Buckets          map[string]int      `json:"buckets"`
	PromptTokens     int                 `json:"prompt_tokens"`
	CachedTokens     int                 `json:"cached_tokens"`
	CompletionTokens int                 `json:"completion_tokens"`
	Summarized       int                 `json:"summarized"`
	Dropped          int                 `json:"dropped"`
	Locations        []agent.Location    `json:"locations"`
	Messages         []agent.ChatMessage `json:"messages"`
}

// runOrLoad reuses a cached trajectory if present, else runs the agent, grades it,
// persists the trajectory, and returns the aggregation record.
func runOrLoad(t *testing.T, ctx context.Context, p runParams) RunRecord {
	t.Helper()
	path := filepath.Join("..", "bench", "runs", "stage3", p.arm, fmt.Sprintf("%s__r%d.json", p.task.ID, p.repeat))
	if data, err := os.ReadFile(path); err == nil {
		var tr stage3Trajectory
		if json.Unmarshal(data, &tr) == nil {
			t.Logf("reuse cached %s / %s r%d", p.arm, p.task.ID, p.repeat)
			return trajToRecord(tr)
		}
	}

	var comp compress.Compressor
	var ms *agent.MeteringSummarizer
	if p.budget > 0 {
		ms = &agent.MeteringSummarizer{Inner: p.compChat, Tok: p.tok}
		comp = &compress.GuidelineCompressor{Summarizer: ms, Tok: p.tok, Guideline: p.guideline}
	}
	deps := agent.Deps{
		Model: p.chat, Tools: p.tools, System: agent.DefaultSystemPrompt,
		MaxTurns: p.maxTurns, Tok: p.tok, Compressor: comp, HistoryBudget: p.budget,
	}
	out, transcript, ledger, runErr := agent.Run(ctx, agent.RunInput{ID: p.task.ID, Query: p.task.Query}, deps)

	buckets := map[string]int{}
	var prompt, cached, completion int
	for _, c := range ledger.Calls {
		for b, v := range c.Buckets {
			buckets[b] += v
		}
		prompt += c.Usage.PromptTokens
		cached += c.Usage.CachedTokens
		completion += c.Usage.CompletionTokens
	}
	if ms != nil {
		// The compressor's own (tiktoken) tokens, charged into the treatment total
		// so the reduction is net of overhead. Its dollar cost is small and lives in
		// this bucket; the net-cost guard prices the agent model, where the prefix-
		// cache effect lives.
		buckets[BucketCompression] += ms.InputTokens + ms.OutputTokens
	}

	recall := Recall(p.task.Gold, locationsToChunks(out.Locations))
	localization := p.task.Bucket == Localization
	tr := stage3Trajectory{
		TaskID: p.task.ID, Arm: p.arm, Budget: p.budget, Repeat: p.repeat, Query: p.task.Query,
		Localization: localization, Success: runErr == nil && localization && recall >= 1.0, Recall: recall,
		StopReason: out.StopReason, Turns: out.Turns, Buckets: buckets,
		PromptTokens: prompt, CachedTokens: cached, CompletionTokens: completion,
		Summarized: out.Summarized, Dropped: out.Dropped,
		Locations: out.Locations, Messages: transcript.Messages,
	}

	// A hard run error (e.g. the model can't produce a valid tool call after the
	// client's retries) is recorded as a FAILED task, not a fatal abort — one bad
	// task must not kill a long, rate-limited run. It is NOT persisted, so a re-run
	// retries it rather than caching the failure.
	if runErr != nil {
		// A bad API key fails every task — abort with one clear message instead of
		// logging 40 doomed runs and printing a misleading success=0 summary.
		if errors.Is(runErr, agent.ErrUnauthorized) {
			t.Fatalf("%v — fix the key and re-run (cached tasks resume; nothing was wasted)", runErr)
		}
		tr.StopReason = "error"
		t.Logf("ERROR %s / %s r%d (not cached, will retry on re-run): %v", p.arm, p.task.ID, p.repeat, runErr)
		return trajToRecord(tr)
	}
	if err := writeJSON(path, tr); err != nil {
		t.Fatalf("persist trajectory %s: %v", path, err)
	}
	t.Logf("%s / %s r%d: success=%v turns=%d stop=%s", p.arm, p.task.ID, p.repeat, tr.Success, tr.Turns, tr.StopReason)
	return trajToRecord(tr)
}

func trajToRecord(tr stage3Trajectory) RunRecord {
	return RunRecord{
		TaskID: tr.TaskID, Localization: tr.Localization, Success: tr.Success, Buckets: tr.Buckets,
		PromptTokens: tr.PromptTokens, CachedTokens: tr.CachedTokens, CompletionTokens: tr.CompletionTokens,
		Summarized: tr.Summarized, Dropped: tr.Dropped,
	}
}

func locationsToChunks(locs []agent.Location) []retrieve.RetrievedChunk {
	out := make([]retrieve.RetrievedChunk, len(locs))
	for i, l := range locs {
		out[i] = retrieve.RetrievedChunk{Path: l.Path, StartLine: l.Start, EndLine: l.End}
	}
	return out
}

func writeJSON(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func parseBudgets(s string) []int {
	var out []int
	for _, p := range strings.Split(s, ",") {
		if n, err := strconv.Atoi(strings.TrimSpace(p)); err == nil && n > 0 {
			out = append(out, n)
		}
	}
	if len(out) == 0 {
		out = []int{3000}
	}
	return out
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}
