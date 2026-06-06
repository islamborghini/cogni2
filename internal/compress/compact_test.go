package compress

import (
	"context"
	"testing"
)

// compactHistory builds a goal + several large observations + a most-recent
// action/observation pair. Total is well over any small budget, so a checkpoint must
// fire and summarize the older observations while leaving the goal and the last pair
// verbatim.
func compactHistory() []Turn {
	// Multi-line so countingSummarizer (keeps the first line) shrinks it; the first
	// line carries the path:line anchor, as a real observation would.
	big := func(n int) string {
		s := "f.py:1-9"
		for i := 0; i < n; i++ {
			s += "\nword"
		}
		return s
	}
	return []Turn{
		{Role: RoleUser, Content: "the goal", Step: 0},
		{Role: RoleAssistant, Content: "act 1", Step: 1},
		{Role: RoleTool, Content: big(40), Step: 2}, // summarizable backlog
		{Role: RoleAssistant, Content: "act 2", Step: 3},
		{Role: RoleTool, Content: big(40), Step: 4},           // summarizable backlog
		{Role: RoleAssistant, Content: "act 3", Step: 5},      // most-recent action (protected)
		{Role: RoleTool, Content: "small.py:1-2 ok", Step: 6}, // most-recent observation (protected)
	}
}

func TestCompactFiresOnceToTarget(t *testing.T) {
	h := compactHistory()
	tok := wordTok{}
	budget := 60
	c := &CompactingCompressor{Summarizer: &countingSummarizer{}, Tok: tok, TargetFraction: 0.5}

	res, err := c.Compress(context.Background(), h, budget)
	if err != nil {
		t.Fatal(err)
	}
	if res.Summarized == 0 {
		t.Fatal("checkpoint should have summarized older observations")
	}
	// Compacts to the low-water target (~half budget), not merely under budget.
	if got := totalTokens(res.History, tok); got > budget/2+5 {
		t.Errorf("did not compact to low-water target: total %d, target ~%d", got, budget/2)
	}
}

// TestCompactIdempotent is the property the per-turn rewrite policy lacked: once a
// checkpoint has compacted to the low-water target, an immediate second call is a
// no-op — no new summarizer calls, identical content — so the cached prefix does not
// diverge again.
func TestCompactIdempotent(t *testing.T) {
	h := compactHistory()
	tok := wordTok{}
	cs := &countingSummarizer{}
	c := &CompactingCompressor{Summarizer: cs, Tok: tok, TargetFraction: 0.5}

	first, err := c.Compress(context.Background(), h, 60)
	if err != nil {
		t.Fatal(err)
	}
	callsAfterFirst := cs.calls

	second, err := c.Compress(context.Background(), first.History, 60)
	if err != nil {
		t.Fatal(err)
	}
	if cs.calls != callsAfterFirst {
		t.Errorf("second call made %d more summarizer calls; checkpoint must be idempotent", cs.calls-callsAfterFirst)
	}
	if second.Summarized != 0 || second.Dropped != 0 {
		t.Errorf("second call edited history: summarized=%d dropped=%d", second.Summarized, second.Dropped)
	}
	for i := range first.History {
		if second.History[i].Content != first.History[i].Content {
			t.Errorf("step %d content changed on idempotent re-call", i)
		}
	}
}

func TestCompactNoOpUnderBudget(t *testing.T) {
	h := compactHistory()
	tok := wordTok{}
	cs := &countingSummarizer{}
	c := &CompactingCompressor{Summarizer: cs, Tok: tok}

	res, err := c.Compress(context.Background(), h, 100000) // budget far above total
	if err != nil {
		t.Fatal(err)
	}
	if cs.calls != 0 {
		t.Errorf("under budget must make zero summarizer calls, made %d", cs.calls)
	}
	if res.Summarized != 0 || res.Dropped != 0 {
		t.Errorf("under budget must not edit history: %+v", res)
	}
}

func TestCompactKeepsGoalAndRecentVerbatim(t *testing.T) {
	h := compactHistory()
	tok := wordTok{}
	c := &CompactingCompressor{Summarizer: &countingSummarizer{}, Tok: tok, TargetFraction: 0.5}

	res, err := c.Compress(context.Background(), h, 60)
	if err != nil {
		t.Fatal(err)
	}
	if res.History[0].Content != "the goal" {
		t.Errorf("goal was altered: %q", res.History[0].Content)
	}
	last := len(res.History) - 1
	if res.History[last].Content != h[last].Content {
		t.Errorf("most-recent observation was altered: %q", res.History[last].Content)
	}
	if totalTokens(res.History, tok) >= totalTokens(h, tok) {
		t.Error("compaction did not reduce tokens")
	}
}
