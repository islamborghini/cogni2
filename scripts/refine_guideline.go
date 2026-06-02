//go:build ignore

// refine_guideline.go is the offline ACON failure-driven loop. It is NOT part of
// the build, CI, or the agent's hot path (the //go:build ignore tag keeps it out
// of `go build ./...`); you run it by hand:
//
//	GROQ_API_KEY=… go run scripts/refine_guideline.go \
//	  -runs bench/runs/stage3 -guideline internal/compress/guideline.txt
//
// For each frozen task where the BASELINE (full-history) run succeeded but a
// TREATMENT (compressed-history) run failed, it sends both transcripts to a
// capable model, asks which dropped information caused the failure, and appends a
// one-line preservation rule to the guideline. This is the contrastive analysis
// at the heart of ACON. You review the diff before re-running the eval — nothing
// here edits the guideline silently inside the loop. Use -dry-run to preview.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/islamborghini/cogni2/internal/agent"
)

// trajectory mirrors the fields the eval persists (bench/eval_endtoend_test.go's
// stage3Trajectory). Re-declared here because that type lives in a test file and
// cannot be imported; only the fields this script reads are kept.
type trajectory struct {
	TaskID   string              `json:"task_id"`
	Arm      string              `json:"arm"`
	Budget   int                 `json:"budget"`
	Query    string              `json:"query"`
	Success  bool                `json:"success"`
	Messages []agent.ChatMessage `json:"messages"`
}

const refineSystemPrompt = `You analyze why an AI coding agent FAILED a code-navigation task once its history
was compressed, when the SAME task SUCCEEDED with full history.

You are given the task goal, the full-history transcript (succeeded), and the
compressed-history transcript (failed). Identify the SPECIFIC information that
compression dropped or garbled and that caused the failure.

Output exactly ONE concise preservation rule as a single imperative line starting
with a verb (e.g. "Keep the exact line numbers of every class definition.").
The rule must be general enough to help other tasks, not hard-coded to this one.
If you cannot identify a clear dropped-information cause, output exactly: NONE`

func main() {
	runsDir := flag.String("runs", "bench/runs/stage3", "directory of persisted Stage 3 trajectories")
	guidelinePath := flag.String("guideline", "internal/compress/guideline.txt", "guideline file to append rules to")
	model := flag.String("model", envOr("COGNI_REFINE_MODEL", "llama-3.3-70b-versatile"), "analysis model")
	dryRun := flag.Bool("dry-run", false, "print proposed rules without editing the guideline")
	flag.Parse()

	trajs := loadTrajectories(*runsDir)
	if len(trajs) == 0 {
		log.Fatalf("no trajectories under %s — run the eval first", *runsDir)
	}
	byTask := map[string][]trajectory{}
	for _, tr := range trajs {
		byTask[tr.TaskID] = append(byTask[tr.TaskID], tr)
	}

	chat, err := agent.FromEnv(*model)
	if err != nil {
		log.Fatalf("model: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	var rules []string
	for taskID, runs := range byTask {
		base, ok := find(runs, func(t trajectory) bool { return t.Arm == "baseline" && t.Success })
		if !ok {
			continue
		}
		fail, ok := find(runs, func(t trajectory) bool { return strings.HasPrefix(t.Arm, "budget-") && !t.Success })
		if !ok {
			continue
		}
		rule, err := askForRule(ctx, chat, base, fail)
		if err != nil {
			log.Printf("task %s: %v", taskID, err)
			continue
		}
		if rule == "" {
			log.Printf("task %s: no clear dropped-info cause", taskID)
			continue
		}
		log.Printf("task %s (failed at budget %d): %s", taskID, fail.Budget, rule)
		rules = append(rules, fmt.Sprintf("- (from %s, budget %d) %s", taskID, fail.Budget, rule))
	}

	if len(rules) == 0 {
		fmt.Println("No baseline-success / compressed-failure pairs found — nothing to add.")
		return
	}
	if *dryRun {
		fmt.Printf("\nProposed rules (dry run, not written):\n%s\n", strings.Join(rules, "\n"))
		return
	}
	if err := appendRules(*guidelinePath, rules); err != nil {
		log.Fatalf("append rules: %v", err)
	}
	fmt.Printf("Appended %d rule(s) to %s. Review the diff before re-running the eval.\n", len(rules), *guidelinePath)
}

func askForRule(ctx context.Context, chat *agent.OpenAIChat, base, fail trajectory) (string, error) {
	var b strings.Builder
	fmt.Fprintf(&b, "TASK GOAL: %s\n\n", base.Query)
	fmt.Fprintf(&b, "=== FULL HISTORY (succeeded) ===\n%s\n", renderTranscript(base.Messages))
	fmt.Fprintf(&b, "=== COMPRESSED HISTORY (failed) ===\n%s\n", renderTranscript(fail.Messages))
	out, err := chat.Generate(ctx, agent.Request{
		System:   refineSystemPrompt,
		Messages: []agent.ChatMessage{{Role: agent.RoleUser, Content: b.String()}},
	})
	if err != nil {
		return "", err
	}
	rule := strings.TrimSpace(out.Message.Content)
	if rule == "" || strings.EqualFold(rule, "NONE") {
		return "", nil
	}
	return firstLine(rule), nil
}

func renderTranscript(msgs []agent.ChatMessage) string {
	var b strings.Builder
	for _, m := range msgs {
		content := m.Content
		if len(content) > 1500 {
			content = content[:1500] + "…(truncated)"
		}
		b.WriteString(m.Role)
		for _, tc := range m.ToolCalls {
			fmt.Fprintf(&b, " [calls %s(%s)]", tc.Name, tc.Args)
		}
		b.WriteString(": ")
		b.WriteString(content)
		b.WriteByte('\n')
	}
	return b.String()
}

func loadTrajectories(dir string) []trajectory {
	var out []trajectory
	_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".json") {
			return nil
		}
		data, rerr := os.ReadFile(path)
		if rerr != nil {
			return nil
		}
		var tr trajectory
		if json.Unmarshal(data, &tr) == nil && tr.TaskID != "" {
			out = append(out, tr)
		}
		return nil
	})
	return out
}

const learnedHeader = "## Learned rules (refined from failure analysis)"

func appendRules(path string, rules []string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	body := string(data)
	var b strings.Builder
	b.WriteString(strings.TrimRight(body, "\n"))
	b.WriteString("\n\n")
	if !strings.Contains(body, learnedHeader) {
		b.WriteString(learnedHeader)
		b.WriteString("\n")
	}
	for _, r := range rules {
		b.WriteString(r)
		b.WriteString("\n")
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func find(ts []trajectory, pred func(trajectory) bool) (trajectory, bool) {
	for _, t := range ts {
		if pred(t) {
			return t, true
		}
	}
	return trajectory{}, false
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[:i])
	}
	return s
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
