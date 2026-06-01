package index

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/islamborghini/cogni2/internal/chunk"
	"github.com/islamborghini/cogni2/internal/embed"
)

// fieldsTok is the offline token counter used across the project's unit tests.
type fieldsTok struct{}

func (fieldsTok) Count(s string) int { return len(strings.Fields(s)) }

// countingEmb wraps an Embedder and records how many texts it has embedded, so a
// test can prove the SHA cache skips re-embedding unchanged chunks.
type countingEmb struct {
	inner embed.Embedder
	calls int
}

func (c *countingEmb) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	c.calls += len(texts)
	return c.inner.Embed(ctx, texts)
}

func writeFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

func newStore(t *testing.T, doc embed.Embedder) *Store {
	t.Helper()
	s, err := Open(":memory:", Config{
		DocEmbedder:   doc,
		QueryEmbedder: embed.Fake{},
		Tokenizer:     fieldsTok{},
		ChunkOptions:  chunk.Options{MaxChunkTokens: 1000, Merge: false},
	})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestStore_BuildRetrieveAndCache(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "a.py", "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n")
	writeFile(t, dir, "b.py", "class Widget:\n    def render(self):\n        return \"<div>\"\n")

	doc := &countingEmb{inner: embed.Fake{}}
	s := newStore(t, doc)
	ctx := context.Background()

	n, err := s.BuildAll(ctx, dir)
	if err != nil {
		t.Fatalf("BuildAll: %v", err)
	}
	if n != 2 {
		t.Errorf("indexed %d files, want 2", n)
	}
	count, _ := s.Count()
	if count == 0 {
		t.Fatal("expected indexed chunks, got 0")
	}
	if doc.calls != count {
		t.Errorf("embedded %d chunks but stored %d", doc.calls, count)
	}

	// Re-indexing unchanged files must not re-embed anything (SHA cache hit).
	before := doc.calls
	if _, err := s.BuildAll(ctx, dir); err != nil {
		t.Fatalf("BuildAll (2nd): %v", err)
	}
	if doc.calls != before {
		t.Errorf("re-index re-embedded %d chunks; cache should have skipped all", doc.calls-before)
	}

	// An exact-content query must rank its own chunk first (cosine == 1).
	all, err := s.Retrieve(ctx, "alpha", 100)
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	var target string
	for _, c := range all {
		if strings.Contains(c.Content, "alpha") {
			target = c.Content
		}
	}
	if target == "" {
		t.Fatal("alpha chunk not indexed")
	}
	top, err := s.Retrieve(ctx, target, 1)
	if err != nil {
		t.Fatalf("Retrieve exact: %v", err)
	}
	if len(top) != 1 || !strings.Contains(top[0].Content, "alpha") {
		t.Errorf("exact-content query did not return its own chunk: %+v", top)
	}
	if top[0].Score < 0.99 {
		t.Errorf("exact match score = %v, want ~1.0", top[0].Score)
	}
}

func TestStore_DeleteFile(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "a.py", "def alpha():\n    return 1\n")
	writeFile(t, dir, "b.py", "def beta():\n    return 2\n")

	s := newStore(t, embed.Fake{})
	ctx := context.Background()
	if _, err := s.BuildAll(ctx, dir); err != nil {
		t.Fatalf("BuildAll: %v", err)
	}
	before, _ := s.Count()

	if err := s.DeleteFile("a.py"); err != nil {
		t.Fatalf("DeleteFile: %v", err)
	}
	after, _ := s.Count()
	if after >= before {
		t.Errorf("delete did not drop rows: before=%d after=%d", before, after)
	}
	res, _ := s.Retrieve(ctx, "anything", 100)
	for _, c := range res {
		if c.Path == "a.py" {
			t.Errorf("deleted file still retrievable: %+v", c)
		}
	}
}
