//go:build eval

package bench

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/islamborghini/cogni2/internal/chunk"
	"github.com/islamborghini/cogni2/internal/embed"
	"github.com/islamborghini/cogni2/internal/index"
	"github.com/islamborghini/cogni2/internal/meter"
)

// evalEnv is the shared setup the gated end-to-end evals need: the built (or
// cached) index, the tokenizer, the frozen task set, and the run metadata.
type evalEnv struct {
	store    *index.Store
	tok      meter.Tokenizer
	set      *TaskSet
	provider string
	model    string
	maxTok   int
	files    int
	chunks   int
}

// setupEvalIndex applies the eval gates, loads the frozen task set, and builds (or
// reuses) the content-addressed index keyed by repo + embedder + chunk budget. The
// store is closed via t.Cleanup. Both TestRetrieval (Stage 1) and TestSkeleton
// (Stage 2) share this so they index the corpus identically and reuse the same
// cached vectors — Stage 2 then costs only the per-task query embeddings.
func setupEvalIndex(t *testing.T) *evalEnv {
	t.Helper()
	if os.Getenv("COGNI_EVAL") != "1" {
		t.Skip("set COGNI_EVAL=1 to run the embedding eval")
	}
	repo := os.Getenv("COGNI_BENCH_REPO")
	if repo == "" {
		t.Skip("set COGNI_BENCH_REPO to a checkout of the target repo")
	}

	set, err := Load("tasks.yaml")
	if err != nil {
		t.Fatalf("load tasks: %v", err)
	}
	tok, err := meter.Default()
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	docEmb, err := embed.FromEnv("document")
	if err != nil {
		t.Fatalf("doc embedder: %v", err)
	}
	queryEmb, err := embed.FromEnv("query")
	if err != nil {
		t.Fatalf("query embedder: %v", err)
	}

	maxTok := chunk.DefaultMaxChunkTokens
	if v := os.Getenv("CHUNK_MAX_TOKENS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxTok = n
		}
	}
	provider := envOrDefault("EMBED_PROVIDER", "voyage")
	model := envOrDefault("EMBED_MODEL", "voyage-code-3")
	t.Logf("chunk budget: %d tokens; embedder %s/%s", maxTok, provider, model)

	cacheDir := os.Getenv("COGNI_INDEX_DIR")
	if cacheDir == "" {
		cacheDir = filepath.Join(os.TempDir(), "cogni2-index")
	}
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		t.Fatalf("mkdir index cache: %v", err)
	}
	key := sha256.Sum256([]byte(fmt.Sprintf("%s|%s|%s|%d", repo, provider, model, maxTok)))
	dbPath := filepath.Join(cacheDir, fmt.Sprintf("index-%x.db", key[:8]))
	t.Logf("index cache: %s", dbPath)

	store, err := index.Open(dbPath, index.Config{
		DocEmbedder:   docEmb,
		QueryEmbedder: queryEmb,
		Tokenizer:     tok,
		ChunkOptions:  chunk.Options{MaxChunkTokens: maxTok, Merge: true},
	})
	if err != nil {
		t.Fatalf("open index: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ctx := context.Background()
	if cached, _ := store.Count(); cached > 0 {
		t.Logf("reusing cached index (%d chunks) — unchanged files are not re-embedded", cached)
	}
	t.Logf("indexing %s …", repo)
	files, err := store.BuildAll(ctx, repo)
	if err != nil {
		t.Fatalf("build index: %v", err)
	}
	chunks, _ := store.Count()
	t.Logf("indexed %d files into %d chunks", files, chunks)

	return &evalEnv{
		store: store, tok: tok, set: set,
		provider: provider, model: model, maxTok: maxTok,
		files: files, chunks: chunks,
	}
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
