//go:build eval

// This is the Stage 1 end-to-end eval: it embeds the real pinned repository with
// Voyage and measures recall@10 + mean retrieved tokens over the frozen task set.
// It is gated behind the `eval` build tag and three env vars so it never runs in
// normal `go test ./...` or in CI — only when explicitly driven with an API key
// and a checkout.
//
//	export COGNI_EVAL=1
//	export COGNI_BENCH_REPO=/path/to/django   # checked out at target_sha
//	# default: Voyage (code-specialized)
//	export VOYAGE_API_KEY=…
//	# or a local, no-quota OpenAI-compatible server (e.g. Ollama):
//	#   export EMBED_PROVIDER=ollama EMBED_MODEL=mxbai-embed-large CHUNK_MAX_TOKENS=400
//	go test -tags eval ./bench/ -run Retrieval -v -timeout 60m
//
// The index is cached under COGNI_INDEX_DIR (default $TMPDIR/cogni2-index), keyed
// by repo + embedder + budget. The first run pays to embed the corpus; later runs
// (more tasks, Stage 2/3) reuse the stored vectors and only embed the cheap task
// queries, so re-measuring is effectively free.
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

const retrievalK = 10

func TestRetrieval(t *testing.T) {
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
	// EMBED_PROVIDER selects the embedder (Voyage by default, or an
	// OpenAI-compatible/Ollama server); each validates its own credentials.
	docEmb, err := embed.FromEnv("document")
	if err != nil {
		t.Fatalf("doc embedder: %v", err)
	}
	queryEmb, err := embed.FromEnv("query")
	if err != nil {
		t.Fatalf("query embedder: %v", err)
	}

	// The budget is in tiktoken tokens; a wordpiece embedder (e.g. Ollama models)
	// tokenizes code more finely, so set CHUNK_MAX_TOKENS lower than the default
	// to keep chunks within its context window.
	maxTok := chunk.DefaultMaxChunkTokens
	if v := os.Getenv("CHUNK_MAX_TOKENS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxTok = n
		}
	}
	provider := envOrDefault("EMBED_PROVIDER", "voyage")
	model := envOrDefault("EMBED_MODEL", "voyage-code-3")
	t.Logf("chunk budget: %d tokens; embedder %s/%s", maxTok, provider, model)

	// Persistent, content-addressed index cache keyed by repo + embedder + budget.
	// Re-running the same config reuses stored vectors (the store's SHA cache
	// skips re-embedding unchanged chunks), so adding tasks / running later stages
	// costs only the cheap per-task query embeddings — not a full re-index.
	// Override the location with COGNI_INDEX_DIR.
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
	defer func() { _ = store.Close() }()

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

	results := make([]TaskResult, 0, len(set.Tasks))
	for _, task := range set.Tasks {
		retrieved, err := store.Retrieve(ctx, task.Query, retrievalK)
		if err != nil {
			t.Fatalf("retrieve %q: %v", task.ID, err)
		}

		m := meter.New(tok, 1, task.ID)
		for _, c := range retrieved {
			m.Add(meter.BucketRetrievedCode, c.Content)
		}
		rec := m.Record()
		if _, err := m.Persist(".."); err != nil { // -> bench/runs/stage1/<id>.json
			t.Fatalf("persist meter %q: %v", task.ID, err)
		}

		results = append(results, TaskResult{
			ID:              task.ID,
			Bucket:          task.Bucket,
			GoldSize:        len(task.Gold),
			Recall:          Recall(task.Gold, retrieved),
			RetrievedTokens: rec.Buckets[meter.BucketRetrievedCode],
		})
	}

	if err := os.MkdirAll("results", 0o755); err != nil {
		t.Fatalf("mkdir results: %v", err)
	}
	run := fmt.Sprintf("- embedder: `%s` / `%s`\n- chunk budget: %d tokens (tiktoken), merge on\n"+
		"- corpus: %d files, %d chunks\n- k: %d\n"+
		"- note: a general-purpose embedder is a floor; `voyage-code-3` is the code-specialized reference.",
		provider, model, maxTok, files, chunks, retrievalK)
	md := RenderMarkdown(set, results, retrievalK, run)
	if err := os.WriteFile(filepath.Join("results", "stage1.md"), []byte(md), 0o644); err != nil {
		t.Fatalf("write stage1.md: %v", err)
	}

	recall, meanTok := Headline(results)
	fmt.Printf("\n=== Stage 1 ===\nrecall@%d (localization) = %.3f\nmean_retrieved_tokens = %.0f\n",
		retrievalK, recall, meanTok)
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
