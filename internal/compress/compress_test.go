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

// countingSummarizer records how many times it is called and shrinks multi-line
// (and single-line) input to "<first line> S", so the tests can assert that
// compression makes the minimum number of summarizer calls.
type countingSummarizer struct{ calls int }

func (c *countingSummarizer) Summarize(_ context.Context, text, _ string) (string, error) {
	c.calls++
	first := text
	if i := strings.IndexByte(text, '\n'); i >= 0 {
		first = text[:i]
	}
	return first + " S", nil
}

func TestCompressNoOpUnderBudget(t *testing.T) {
	// History already fits the budget → no summarizer call at all. This is the fix
	// that stops compression overhead from dwarfing the savings on cheap tasks.
	history := []Turn{
		{Role: RoleUser, Content: "find the thing", Step: 0},
		{Role: RoleTool, Content: "a.py:1-2\nsome body here", Step: 0},
		{Role: RoleAssistant, Content: "reading", Step: 1},
		{Role: RoleTool, Content: "b.py:1-2 fresh", Step: 1},
	}
	cs := &countingSummarizer{}
	c := &GuidelineCompressor{Summarizer: cs, Tok: wordTok{}}

	res, err := c.Compress(context.Background(), history, 1000)
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if cs.calls != 0 || res.Summarized != 0 || res.Dropped != 0 {
		t.Fatalf("under budget must be a no-op: calls=%d res=%+v", cs.calls, res)
	}
	if res.History[1].Content != history[1].Content {
		t.Errorf("content changed under budget: %q", res.History[1].Content)
	}
}

func TestCompressSummarizesOldestUntilFit(t *testing.T) {
	history := []Turn{
		{Role: RoleUser, Content: "goal", Step: 0},                              // 1 (protected goal)
		{Role: RoleTool, Content: "a.py:1-2\nbig big big big big big", Step: 0}, // 7
		{Role: RoleTool, Content: "b.py:1-2\nbig big big big big big", Step: 1}, // 7
		{Role: RoleAssistant, Content: "act now", Step: 2},                      // 2 (protected)
		{Role: RoleTool, Content: "c.py:1-2 fresh", Step: 2},                    // 2 (protected recent)
	}
	cs := &countingSummarizer{}
	c := &GuidelineCompressor{Summarizer: cs, Tok: wordTok{}}

	res, err := c.Compress(context.Background(), history, 15) // total 19 → summarize one
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if res.Summarized != 1 || cs.calls != 1 || res.Dropped != 0 {
		t.Fatalf("want exactly one summarize: summarized=%d calls=%d dropped=%d", res.Summarized, cs.calls, res.Dropped)
	}
	out := res.History
	if out[1].Content != "a.py:1-2 S" || !out[1].Compressed {
		t.Errorf("oldest obs not summarized/marked: %+v", out[1])
	}
	if out[2].Content != history[2].Content || out[2].Compressed {
		t.Errorf("second obs should be untouched: %+v", out[2])
	}
	if out[0].Content != "goal" || out[4].Content != "c.py:1-2 fresh" {
		t.Errorf("goal or most-recent observation was mutated")
	}
}

func TestCompressIncrementalDoesNotResummarize(t *testing.T) {
	history := []Turn{
		{Role: RoleUser, Content: "goal", Step: 0},
		{Role: RoleTool, Content: "a.py:1-2\nbig big big big big big", Step: 0},
		{Role: RoleTool, Content: "b.py:1-2\nbig big big big big big", Step: 1},
		{Role: RoleAssistant, Content: "act now", Step: 2},
		{Role: RoleTool, Content: "c.py:1-2 fresh", Step: 2},
	}
	c := &GuidelineCompressor{Summarizer: &countingSummarizer{}, Tok: wordTok{}}
	res1, err := c.Compress(context.Background(), history, 15) // summarizes obs1 only
	if err != nil {
		t.Fatalf("compress: %v", err)
	}

	// Second pass over the already-partly-compressed history with a tighter budget:
	// obs1 is marked Compressed and must NOT be summarized again — only obs2.
	c2 := &countingSummarizer{}
	c.Summarizer = c2
	res2, err := c.Compress(context.Background(), res1.History, 10) // total 14 → summarize obs2
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if c2.calls != 1 || res2.Summarized != 1 {
		t.Fatalf("second pass should summarize only the newly-aged obs: calls=%d summarized=%d", c2.calls, res2.Summarized)
	}
	if res2.History[1].Content != "a.py:1-2 S" {
		t.Errorf("already-compressed obs1 was re-summarized: %q", res2.History[1].Content)
	}
	if res2.History[2].Content != "b.py:1-2 S" || !res2.History[2].Compressed {
		t.Errorf("obs2 should now be summarized and marked: %+v", res2.History[2])
	}
}

func TestCompressDropsWhenSummarizingNotEnough(t *testing.T) {
	history := []Turn{
		{Role: RoleUser, Content: "goal", Step: 0},
		{Role: RoleTool, Content: "a.py:1-2 big big big big big big", Step: 0}, // 7, single line
		{Role: RoleTool, Content: "b.py:1-2 big big big big big big", Step: 1}, // 7
		{Role: RoleAssistant, Content: "act now", Step: 2},
		{Role: RoleTool, Content: "c.py:1-2 fresh", Step: 2},
	}
	// MinSummarizeTokens high → nothing is summarized, so dropping is the only lever.
	c := &GuidelineCompressor{Summarizer: &countingSummarizer{}, Tok: wordTok{}, MinSummarizeTokens: 1000}

	res, err := c.Compress(context.Background(), history, 16) // total 19 → drop oldest
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if res.Summarized != 0 || res.Dropped < 1 {
		t.Fatalf("want a drop, not a summarize: %+v", res)
	}
	if res.History[1].Content != dropMarker {
		t.Errorf("oldest observation should be dropped: %q", res.History[1].Content)
	}
	if res.History[4].Content != "c.py:1-2 fresh" {
		t.Errorf("most recent observation must never be dropped")
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
