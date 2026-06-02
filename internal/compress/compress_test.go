package compress

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// wordTok is a deterministic, offline tokenizer for the tests: one token per
// whitespace-separated word. The real tiktoken counter lives in meter; the
// compressor only needs a Tokenizer, so the tests inject this and stay hermetic.
type wordTok struct{}

func (wordTok) Count(s string) int { return len(strings.Fields(s)) }

func TestCompressKeepsGoalAndRecentVerbatim(t *testing.T) {
	history := []Turn{
		{Role: RoleUser, Content: "find slugify implementation", Step: 0},
		{Role: RoleAssistant, Content: "I will search the codebase.", Step: 1},
		{Role: RoleTool, Content: "django/utils/text.py:1-5\nold body\nmore old words here", Step: 1, Kind: "retrieved_code"},
		{Role: RoleAssistant, Content: "Now I will read the file.", Step: 2},
		{Role: RoleTool, Content: "django/utils/text.py:40-50\nfresh body content", Step: 2},
	}
	c := &GuidelineCompressor{Summarizer: FakeSummarizer{}, Tok: wordTok{}}

	res, err := c.Compress(context.Background(), history, 1000)
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if res.Summarized != 1 || res.Dropped != 0 {
		t.Fatalf("summarized=%d dropped=%d, want 1 and 0", res.Summarized, res.Dropped)
	}
	out := res.History
	if out[0].Content != "find slugify implementation" {
		t.Errorf("goal was mutated: %q", out[0].Content)
	}
	if out[1].Content != "I will search the codebase." {
		t.Errorf("assistant turn was mutated: %q", out[1].Content)
	}
	if !strings.Contains(out[2].Content, "django/utils/text.py:1-5") {
		t.Errorf("older observation lost its anchor: %q", out[2].Content)
	}
	if c.Tok.Count(out[2].Content) >= c.Tok.Count(history[2].Content) {
		t.Errorf("older observation was not shortened: %q", out[2].Content)
	}
	if out[4].Content != history[4].Content {
		t.Errorf("most recent observation was mutated: %q", out[4].Content)
	}
	// The input slice must not be aliased/mutated.
	if history[2].Content == out[2].Content {
		t.Error("Compress mutated the caller's slice")
	}
}

func TestCompressDropsOldestFirstOverBudget(t *testing.T) {
	history := []Turn{
		{Role: RoleUser, Content: "goal here", Step: 0},
		{Role: RoleTool, Content: "a.py:1-2 alpha beta gamma delta epsilon zeta eta", Step: 0},
		{Role: RoleTool, Content: "b.py:1-2 alpha beta gamma delta epsilon zeta eta", Step: 1},
		{Role: RoleAssistant, Content: "acting now", Step: 2},
		{Role: RoleTool, Content: "c.py:1-2 fresh stuff", Step: 2},
	}
	// Skip summarization (MinSummarizeTokens huge) to test the drop pass in
	// isolation: raw observations totalling 23 tokens, budget 18, so exactly the
	// oldest observation must be evicted.
	c := &GuidelineCompressor{Summarizer: FakeSummarizer{}, Tok: wordTok{}, MinSummarizeTokens: 1000}

	res, err := c.Compress(context.Background(), history, 18)
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if res.Summarized != 0 {
		t.Errorf("summarized=%d, want 0 (summarization skipped)", res.Summarized)
	}
	if res.Dropped != 1 {
		t.Fatalf("dropped=%d, want exactly 1 (oldest only)", res.Dropped)
	}
	out := res.History
	if out[1].Content != dropMarker {
		t.Errorf("oldest observation should be the drop marker, got %q", out[1].Content)
	}
	if out[2].Content != history[2].Content {
		t.Errorf("second observation should be untouched, got %q", out[2].Content)
	}
	if out[4].Content != history[4].Content {
		t.Errorf("most recent observation must never be dropped, got %q", out[4].Content)
	}
	if got := c.total(out); got > 18 {
		t.Errorf("history still over budget after drops: %d > 18", got)
	}
}

func TestCompressRejectsNonShrinkingSummary(t *testing.T) {
	// A short single-line observation: the FakeSummarizer would only append a
	// marker, making it longer, so it must be left verbatim.
	history := []Turn{
		{Role: RoleUser, Content: "goal", Step: 0},
		{Role: RoleTool, Content: "x.py:1-2 ok", Step: 0},
		{Role: RoleAssistant, Content: "act", Step: 1},
		{Role: RoleTool, Content: "y.py:1-2 fresh", Step: 1},
	}
	c := &GuidelineCompressor{Summarizer: FakeSummarizer{}, Tok: wordTok{}}

	res, err := c.Compress(context.Background(), history, 1000)
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if res.Summarized != 0 {
		t.Errorf("summarized=%d, want 0 (summary would not shrink)", res.Summarized)
	}
	if res.History[1].Content != "x.py:1-2 ok" {
		t.Errorf("short observation was changed: %q", res.History[1].Content)
	}
}

func TestCompressEmptyHistory(t *testing.T) {
	c := &GuidelineCompressor{Summarizer: FakeSummarizer{}, Tok: wordTok{}}
	res, err := c.Compress(context.Background(), nil, 100)
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if len(res.History) != 0 || res.Summarized != 0 || res.Dropped != 0 {
		t.Errorf("empty history should be a no-op, got %+v", res)
	}
}

func TestProtectedMask(t *testing.T) {
	turns := []Turn{
		{Role: RoleUser}, {Role: RoleAssistant}, {Role: RoleTool}, {Role: RoleAssistant}, {Role: RoleTool},
	}
	got := protectedMask(turns)
	want := []bool{true, false, false, true, true} // goal + last action onward
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("protectedMask[%d] = %v, want %v (%v)", i, got[i], want[i], got)
			break
		}
	}

	// No assistant and no user → nothing protected.
	none := protectedMask([]Turn{{Role: RoleTool}, {Role: RoleTool}})
	for i, p := range none {
		if p {
			t.Errorf("protectedMask with no goal/action protected index %d", i)
		}
	}
}

func TestFakeSummarizerDeterministicAndShrinks(t *testing.T) {
	in := "path/to/file.py:10-20\nline two\nline three\nline four"
	a, _ := FakeSummarizer{}.Summarize(context.Background(), in, "")
	b, _ := FakeSummarizer{}.Summarize(context.Background(), in, "")
	if a != b {
		t.Errorf("FakeSummarizer is not deterministic: %q vs %q", a, b)
	}
	if !strings.HasPrefix(a, "path/to/file.py:10-20") {
		t.Errorf("FakeSummarizer dropped the leading anchor: %q", a)
	}
	tok := wordTok{}
	if tok.Count(a) >= tok.Count(in) {
		t.Errorf("FakeSummarizer did not shrink a multi-line input: %q", a)
	}
}

func TestGuidelineLoad(t *testing.T) {
	def := DefaultGuideline()
	if !strings.Contains(def, "PRESERVE") {
		t.Errorf("embedded guideline missing PRESERVE rules:\n%s", def)
	}

	got, err := LoadGuideline("")
	if err != nil || got != def {
		t.Errorf("LoadGuideline(\"\") = (%q, %v), want the embedded default", got, err)
	}

	path := filepath.Join(t.TempDir(), "g.txt")
	if err := os.WriteFile(path, []byte("custom rule"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err = LoadGuideline(path)
	if err != nil || got != "custom rule" {
		t.Errorf("LoadGuideline(file) = (%q, %v), want the file contents", got, err)
	}

	if _, err := LoadGuideline(filepath.Join(t.TempDir(), "missing.txt")); err == nil {
		t.Error("LoadGuideline(missing) should return an error")
	}
}
