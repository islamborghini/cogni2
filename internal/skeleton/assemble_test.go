package skeleton

import (
	"fmt"
	"strings"
	"testing"

	"github.com/islamborghini/cogni2/internal/retrieve"
)

// wordTok is a deterministic Tokenizer for assembly tests: one token per
// whitespace-separated field. Block sizes are then easy to reason about.
type wordTok struct{}

func (wordTok) Count(s string) int {
	if s == "" {
		return 0
	}
	return len(strings.Fields(s))
}

// pyFunc builds a valid Python function with bodyLines body statements, so the
// skeletonizer actually skeletonizes it (rather than passing it through).
func pyFunc(name string, bodyLines int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "def %s():\n", name)
	for i := 0; i < bodyLines; i++ {
		fmt.Fprintf(&b, "    %s_%d = %d\n", name, i, i)
	}
	fmt.Fprintf(&b, "    return %s_0\n", name)
	return b.String()
}

func testChunks() []retrieve.RetrievedChunk {
	return []retrieve.RetrievedChunk{
		{Path: "a.py", StartLine: 1, EndLine: 32, Kind: retrieve.KindFunction, Content: pyFunc("alpha", 30), Score: 0.9},
		{Path: "b.py", StartLine: 1, EndLine: 10, Kind: retrieve.KindFunction, Content: pyFunc("beta", 8), Score: 0.8},
		{Path: "c.py", StartLine: 1, EndLine: 10, Kind: retrieve.KindFunction, Content: pyFunc("gamma", 8), Score: 0.7},
	}
}

func blockTokens(c retrieve.RetrievedChunk, tok Tokenizer) int {
	return tok.Count(c.Header() + "\n" + c.Content)
}

func TestAssembleAllFullUnderGenerousBudget(t *testing.T) {
	tok := wordTok{}
	chunks := testChunks()
	a, err := Assemble(chunks, 100000, tok)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if a.FullCount != 3 || a.SkeletonCount != 0 || a.Dropped != 0 {
		t.Fatalf("got full=%d skel=%d dropped=%d, want 3/0/0", a.FullCount, a.SkeletonCount, a.Dropped)
	}
	for _, c := range chunks {
		if !strings.Contains(a.Text, c.Header()) {
			t.Errorf("assembled text missing header %q", c.Header())
		}
	}
	for _, marker := range []string{"alpha_0", "beta_0", "gamma_0"} {
		if !strings.Contains(a.Text, marker) {
			t.Errorf("full body missing marker %q", marker)
		}
	}
}

func TestAssembleSkeletonizesTail(t *testing.T) {
	tok := wordTok{}
	chunks := testChunks()
	total := 0
	for _, c := range chunks {
		total += blockTokens(c, tok)
	}
	// fullBudget = 0.6*total → the top chunk stays full, the tail is skeletonized.
	a, err := Assemble(chunks, total, tok)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if a.SkeletonCount == 0 {
		t.Fatalf("expected some skeletons, got full=%d skel=%d", a.FullCount, a.SkeletonCount)
	}
	if a.Dropped != 0 {
		t.Errorf("expected no drops at total budget, dropped=%d", a.Dropped)
	}
	if !strings.Contains(a.Text, "body omitted") {
		t.Errorf("expected a skeleton placeholder in:\n%s", a.Text)
	}
	// The lowest-ranked chunk is skeletonized: its body vars are gone, anchor stays.
	if strings.Contains(a.Text, "gamma_1") {
		t.Errorf("lowest-ranked body should be omitted:\n%s", a.Text)
	}
	if a.KeptFullTokens < tok.Count(a.Text) {
		t.Errorf("KeptFullTokens (%d) should be >= assembled tokens (%d) — skeletonizing never grows the kept set",
			a.KeptFullTokens, tok.Count(a.Text))
	}
}

func TestAssembleDropsLowestSkeletonsWhenOverBudget(t *testing.T) {
	tok := wordTok{}
	chunks := testChunks()
	budget := blockTokens(chunks[0], tok) // only the top chunk fits
	a, err := Assemble(chunks, budget, tok)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if a.FullCount != 1 {
		t.Errorf("want 1 full body, got %d", a.FullCount)
	}
	if a.Dropped != len(chunks)-1 {
		t.Errorf("want %d dropped, got %d", len(chunks)-1, a.Dropped)
	}
	for _, gone := range []string{"beta", "gamma"} {
		if strings.Contains(a.Text, gone) {
			t.Errorf("dropped chunk %q should not appear:\n%s", gone, a.Text)
		}
	}
	if !strings.Contains(a.Text, "alpha_0") {
		t.Errorf("top chunk should remain full:\n%s", a.Text)
	}
}

func TestAssembleEmpty(t *testing.T) {
	a, err := Assemble(nil, 1000, wordTok{})
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if a.Text != "" || a.FullCount != 0 || a.SkeletonCount != 0 || a.Dropped != 0 {
		t.Errorf("empty input should yield a zero Assembly, got %+v", a)
	}
}
