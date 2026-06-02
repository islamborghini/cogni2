//go:build eval

// This is the Stage 1 end-to-end eval: it embeds the real pinned repository and
// measures recall@10 + mean retrieved tokens over the frozen task set. It is gated
// behind the `eval` build tag and the COGNI_EVAL / COGNI_BENCH_REPO env vars (see
// setupEvalIndex in eval_shared_test.go) so it never runs in normal `go test ./...`
// or CI — only when explicitly driven with an API key and a checkout.
//
//	export COGNI_EVAL=1
//	export COGNI_BENCH_REPO=/path/to/django   # checked out at target_sha
//	export VOYAGE_API_KEY=…                    # or EMBED_PROVIDER=ollama EMBED_MODEL=mxbai-embed-large CHUNK_MAX_TOKENS=400
//	go test -tags eval ./bench/ -run Retrieval -v -timeout 60m
//
// The index is cached (keyed by repo + embedder + budget); the first run pays to
// embed the corpus, later runs (more tasks, Stage 2) reuse the stored vectors.
package bench

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/islamborghini/cogni2/internal/meter"
)

const retrievalK = 10

func TestRetrieval(t *testing.T) {
	env := setupEvalIndex(t)
	ctx := context.Background()

	results := make([]TaskResult, 0, len(env.set.Tasks))
	for _, task := range env.set.Tasks {
		retrieved, err := env.store.Retrieve(ctx, task.Query, retrievalK)
		if err != nil {
			t.Fatalf("retrieve %q: %v", task.ID, err)
		}

		m := meter.New(env.tok, 1, task.ID)
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
		env.provider, env.model, env.maxTok, env.files, env.chunks, retrievalK)
	md := RenderMarkdown(env.set, results, retrievalK, run)
	if err := os.WriteFile(filepath.Join("results", "stage1.md"), []byte(md), 0o644); err != nil {
		t.Fatalf("write stage1.md: %v", err)
	}

	recall, meanTok := Headline(results)
	fmt.Printf("\n=== Stage 1 ===\nrecall@%d (localization) = %.3f\nmean_retrieved_tokens = %.0f\n",
		retrievalK, recall, meanTok)
}
