package bench

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/islamborghini/cogni2/internal/meter"
)

func TestRenderMarkdownStage2(t *testing.T) {
	set := &TaskSet{TargetRepo: "github.com/django/django", TargetSHA: "abc123"}
	res := Stage2Result{
		Recall:                0.824,
		SkeletonsParsed:       50,
		SkeletonParseFailures: 0,
		Rows: []BudgetRow{
			{Budget: 6000, MeanTokens: 3100, SkelReductionPct: 12.5, TotalReductionPct: 12.8, ChunksDropped: 0, TasksWithDrops: 0},
			{Budget: 2000, MeanTokens: 1900, SkelReductionPct: 41.0, TotalReductionPct: 46.5, ChunksDropped: 7, TasksWithDrops: 3},
		},
	}
	md := RenderMarkdownStage2(set, res, 10, "- embedder: `voyage` / `voyage-code-3`")
	for _, want := range []string{
		"Stage 2", "abc123", "tokens down, retrieval and syntax intact",
		"0.824", "50/50", "skeletonization reduction", "total reduction @ budget",
		"6000", "2000", "chunks_dropped", "voyage-code-3",
		"eviction boundary", // boundary present because budget 2000 dropped chunks
		"parse-validity + chunks_dropped",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("stage2 report missing %q\n%s", want, md)
		}
	}
}

func TestStage2DropBoundary(t *testing.T) {
	res := Stage2Result{Rows: []BudgetRow{
		{Budget: 6000, ChunksDropped: 0},
		{Budget: 4000, ChunksDropped: 0},
		{Budget: 3000, ChunksDropped: 2},
		{Budget: 2000, ChunksDropped: 9},
	}}
	if got := res.DropBoundaryBudget(); got != 3000 {
		t.Errorf("DropBoundaryBudget = %d, want 3000 (highest budget that drops)", got)
	}
	none := Stage2Result{Rows: []BudgetRow{{Budget: 6000, ChunksDropped: 0}}}
	if got := none.DropBoundaryBudget(); got != 0 {
		t.Errorf("DropBoundaryBudget (no drops) = %d, want 0", got)
	}
}

func TestLoadStage1Tokens(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "bench", "runs", "stage1")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	rec := meter.Record{
		TaskID:  "alpha",
		Stage:   1,
		Buckets: map[string]int{meter.BucketRetrievedCode: 1234},
		Total:   1234,
	}
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "alpha.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := loadStage1Tokens(root)
	if err != nil {
		t.Fatalf("loadStage1Tokens: %v", err)
	}
	if got["alpha"] != 1234 {
		t.Errorf("alpha retrieved_code tokens = %d, want 1234", got["alpha"])
	}

	// A missing runs directory is an error (Stage 1 not yet run), not a panic.
	if _, err := loadStage1Tokens(t.TempDir()); err == nil {
		t.Error("expected an error when the stage1 runs dir is absent")
	}
}
