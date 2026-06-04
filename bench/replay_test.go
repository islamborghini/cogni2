package bench

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/islamborghini/cogni2/internal/agent"
	"github.com/islamborghini/cogni2/internal/compress"
)

type replayWordTok struct{}

func (replayWordTok) Count(s string) int { return len(strings.Fields(s)) }

// buildTrajectory makes a recorded log with one big early observation followed by
// several cheap turns — the shape where compression wins, because the big
// observation is re-sent (and summarized once) across many turns.
func buildTrajectory() []agent.ChatMessage {
	big := "big.py:1-99\n" + strings.Repeat("token ", 80)
	msgs := []agent.ChatMessage{{Role: agent.RoleUser, Content: "the goal"}}
	add := func(name, args, obs string) {
		msgs = append(msgs,
			agent.ChatMessage{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: name, Name: name, Args: args}}},
			agent.ChatMessage{Role: agent.RoleTool, ToolCallID: name, Content: obs, Origin: "retrieved_code"},
		)
	}
	add("search_code", `{"query":"x"}`, big) // turn 0: the big observation
	for i := 0; i < 4; i++ {                 // turns 1-4: cheap follow-ups
		add("search_code", `{"query":"more"}`, fmt.Sprintf("s%d.py:1-2 small bit", i))
	}
	msgs = append(msgs,
		agent.ChatMessage{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "submit_answer", Name: "submit_answer", Args: `{"locations":[]}`}}},
		agent.ChatMessage{Role: agent.RoleTool, ToolCallID: "submit_answer", Content: "answer recorded", Origin: "history"},
	)
	return msgs
}

func TestReplayCostCompressionNetReducesTokens(t *testing.T) {
	msgs := buildTrajectory()
	tok := replayWordTok{}

	unc, err := ReplayCost(context.Background(), msgs, 2, nil, 0, tok)
	if err != nil {
		t.Fatal(err)
	}
	if unc.Overhead != 0 || unc.Input <= 0 {
		t.Fatalf("uncompressed should have input and no overhead: %+v", unc)
	}

	ms := &agent.MeteringSummarizer{Inner: SizeSummarizer{MaxWords: 4}, Tok: tok}
	comp := &compress.GuidelineCompressor{Summarizer: ms, Tok: tok}
	cmp, err := ReplayCost(context.Background(), msgs, 2, comp, 20, tok)
	if err != nil {
		t.Fatal(err)
	}
	cmp.Overhead = ms.InputTokens + ms.OutputTokens

	if !cmp.Engaged {
		t.Fatal("compression should have engaged on the big observation")
	}
	if cmp.Total() >= unc.Total() {
		t.Fatalf("compressed total %d should be < uncompressed %d (net of overhead %d)", cmp.Total(), unc.Total(), cmp.Overhead)
	}
	if cmp.Output != unc.Output {
		t.Errorf("output must be identical across arms: %d vs %d", cmp.Output, unc.Output)
	}
}

func TestSizeSummarizerShrinksKeepsAnchor(t *testing.T) {
	in := "path/x.py:10-40\n" + strings.Repeat("body ", 50)
	out, _ := SizeSummarizer{MaxWords: 5}.Summarize(context.Background(), in, "")
	if !strings.HasPrefix(out, "path/x.py:10-40") {
		t.Errorf("lost anchor: %q", out)
	}
	tok := replayWordTok{}
	if tok.Count(out) >= tok.Count(in) {
		t.Errorf("did not shrink: %q", out)
	}
}

func TestRenderReplayStage3(t *testing.T) {
	set := &TaskSet{TargetRepo: "github.com/django/django", TargetSHA: "abc123"}
	rows := []ReplayRow{
		{Budget: 2000, MeanTotal: 9500, NetTokenPct: -7.5, GrossTokenPct: 12.5, CostRedPct: 19.4, MeanOverhead: 2600, TasksEngaged: 9, Tasks: 20},
		{Budget: 1000, MeanTotal: 9200, NetTokenPct: -14.6, GrossTokenPct: 13.1, CostRedPct: 20.0, MeanOverhead: 3200, TasksEngaged: 11, Tasks: 20},
	}
	md := RenderReplayStage3(set, 11000, rows, "- agent trajectories: 20")
	for _, want := range []string{
		"Stage 3", "abc123", "by construction", "uncompressed mean total", "11000",
		"net cost", "gross tokens", "net tokens", "20.0%", "2000", "1000",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("replay report missing %q\n%s", want, md)
		}
	}
}
