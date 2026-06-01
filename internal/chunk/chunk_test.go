package chunk

import (
	"fmt"
	"strings"
	"testing"

	"github.com/islamborghini/cogni2/internal/retrieve"
)

// fieldsTok is a deterministic, offline stand-in for the real tiktoken counter:
// one token per whitespace-separated field. Good enough to exercise the size
// budget without a network round-trip.
type fieldsTok struct{}

func (fieldsTok) Count(s string) int { return len(strings.Fields(s)) }

const sampleSrc = `"""Module."""
import os

def alpha():
    return 1

def beta():
    return 2

class Greeter:
    """Greets."""
    LABEL = "hi"

    def greet(self, name):
        return name

    def shout(self, name):
        return name.upper()
`

// find returns the first chunk matching kind and start line, or fails.
func find(t *testing.T, chunks []Chunk, kind string, start int) Chunk {
	t.Helper()
	for _, c := range chunks {
		if c.Kind == kind && c.StartLine == start {
			return c
		}
	}
	t.Fatalf("no %s chunk starting at line %d; got %v", kind, start, summarize(chunks))
	return Chunk{}
}

func summarize(chunks []Chunk) []string {
	out := make([]string, len(chunks))
	for i, c := range chunks {
		out[i] = c.Kind + ":" + itoa(c.StartLine) + "-" + itoa(c.EndLine)
	}
	return out
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [12]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

func TestFile_KindsAndSpans(t *testing.T) {
	chunks, err := File("greeter.py", []byte(sampleSrc), fieldsTok{}, Options{MaxChunkTokens: 1000})
	if err != nil {
		t.Fatalf("File: %v", err)
	}

	alpha := find(t, chunks, retrieve.KindFunction, 4)
	if alpha.EndLine != 5 || !strings.Contains(alpha.Content, "return 1") {
		t.Errorf("alpha chunk wrong: %+v", alpha)
	}
	find(t, chunks, retrieve.KindFunction, 7) // beta
	greet := find(t, chunks, retrieve.KindMethod, 14)
	if greet.EndLine != 15 || !strings.Contains(greet.Content, "return name") {
		t.Errorf("greet method chunk wrong: %+v", greet)
	}
	find(t, chunks, retrieve.KindMethod, 17) // shout

	// Module-level imports/constants are not indexed.
	for _, c := range chunks {
		if strings.Contains(c.Content, "import os") && c.Kind != retrieve.KindClassHeader {
			t.Errorf("import unexpectedly indexed: %+v", c)
		}
	}
}

func TestFile_ClassHeaderIsSignaturesOnly(t *testing.T) {
	chunks, err := File("greeter.py", []byte(sampleSrc), fieldsTok{}, Options{MaxChunkTokens: 1000})
	if err != nil {
		t.Fatalf("File: %v", err)
	}
	hdr := find(t, chunks, retrieve.KindClassHeader, 10)

	for _, want := range []string{"class Greeter:", `LABEL = "hi"`, "def greet(self, name) ...", "def shout(self, name) ..."} {
		if !strings.Contains(hdr.Content, want) {
			t.Errorf("class_header missing %q\n--- content ---\n%s", want, hdr.Content)
		}
	}
	// Method bodies must NOT bleed into the header.
	for _, bad := range []string{"return name", "name.upper()"} {
		if strings.Contains(hdr.Content, bad) {
			t.Errorf("class_header leaked method body %q\n--- content ---\n%s", bad, hdr.Content)
		}
	}
}

func TestFile_MergeSameKind(t *testing.T) {
	noMerge, _ := File("g.py", []byte(sampleSrc), fieldsTok{}, Options{MaxChunkTokens: 1000, Merge: false})
	merged, _ := File("g.py", []byte(sampleSrc), fieldsTok{}, Options{MaxChunkTokens: 1000, Merge: true})

	if len(merged) >= len(noMerge) {
		t.Fatalf("merge did not reduce chunk count: no-merge=%d merged=%d", len(noMerge), len(merged))
	}
	// alpha (4-5) and beta (7-8) are adjacent same-kind funcs → one chunk 4-8.
	ab := find(t, merged, retrieve.KindFunction, 4)
	if ab.EndLine != 8 || !strings.Contains(ab.Content, "return 1") || !strings.Contains(ab.Content, "return 2") {
		t.Errorf("alpha+beta not merged: %+v", ab)
	}
}

func TestFile_ClassHeaderRespectsBudget(t *testing.T) {
	var sb strings.Builder
	sb.WriteString("class Big:\n    \"\"\"A class with very many methods.\"\"\"\n")
	for i := 0; i < 300; i++ {
		fmt.Fprintf(&sb, "    def method_%d(self, a, b, c):\n        return a\n", i)
	}
	chunks, err := File("big.py", []byte(sb.String()), fieldsTok{}, Options{MaxChunkTokens: 40})
	if err != nil {
		t.Fatalf("File: %v", err)
	}
	var sawHeader bool
	for _, c := range chunks {
		if c.Kind != retrieve.KindClassHeader {
			continue
		}
		sawHeader = true
		if n := (fieldsTok{}).Count(c.Content); n > 40 {
			t.Errorf("class_header is %d tokens, exceeds budget 40", n)
		}
		if !strings.Contains(c.Content, "class Big") {
			t.Errorf("class_header dropped its declaration line: %q", c.Content)
		}
	}
	if !sawHeader {
		t.Fatal("no class_header chunk emitted")
	}
}

func TestFile_OversizeSplits(t *testing.T) {
	src := `def big():
    a = 1
    b = 2
    c = 3
    d = 4
    e = 5
    f = 6
`
	chunks, err := File("big.py", []byte(src), fieldsTok{}, Options{MaxChunkTokens: 8})
	if err != nil {
		t.Fatalf("File: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected oversize function to split, got %d chunk(s)", len(chunks))
	}
	for _, c := range chunks {
		if c.Kind != retrieve.KindFunction {
			t.Errorf("split sub-chunk has wrong kind %q", c.Kind)
		}
		if n := (fieldsTok{}).Count(c.Content); n > 8 {
			t.Errorf("sub-chunk exceeds budget (%d tokens): %q", n, c.Content)
		}
	}
}
