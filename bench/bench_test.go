package bench

import (
	"strings"
	"testing"

	"github.com/islamborghini/cogni2/internal/retrieve"
)

func chunkAt(path string, start, end int) retrieve.RetrievedChunk {
	return retrieve.RetrievedChunk{Path: path, StartLine: start, EndLine: end, Kind: retrieve.KindFunction}
}

func TestRecall_SpanOverlap(t *testing.T) {
	gold := []Gold{
		{Path: "a.py", Start: 10, End: 20}, // hit by an overlapping chunk
		{Path: "b.py", Start: 5, End: 5},   // hit exactly
		{Path: "c.py", Start: 100, End: 110},
	}
	retrieved := []retrieve.RetrievedChunk{
		chunkAt("a.py", 15, 30), // overlaps gold[0]
		chunkAt("b.py", 1, 8),   // contains gold[1]
		chunkAt("c.py", 1, 50),  // does NOT reach gold[2]
	}
	if got := Recall(gold, retrieved); got != 2.0/3.0 {
		t.Errorf("recall = %v, want %v", got, 2.0/3.0)
	}
	if got := Recall(nil, retrieved); got != 0 {
		t.Errorf("empty gold recall = %v, want 0", got)
	}
}

func TestRecall_WrongFileIsMiss(t *testing.T) {
	gold := []Gold{{Path: "a.py", Start: 10, End: 20}}
	retrieved := []retrieve.RetrievedChunk{chunkAt("other.py", 10, 20)}
	if got := Recall(gold, retrieved); got != 0 {
		t.Errorf("recall = %v, want 0 (path mismatch)", got)
	}
}

func TestHeadline_LocalizationOnly(t *testing.T) {
	results := []TaskResult{
		{ID: "loc1", Bucket: Localization, Recall: 1.0, RetrievedTokens: 100},
		{ID: "loc2", Bucket: Localization, Recall: 0.5, RetrievedTokens: 200},
		{ID: "enum1", Bucket: Enumeration, Recall: 0.1, RetrievedTokens: 300}, // excluded from recall
	}
	recall, meanTok := Headline(results)
	if recall != 0.75 { // (1.0 + 0.5) / 2, enumeration excluded
		t.Errorf("headline recall = %v, want 0.75", recall)
	}
	if meanTok != 200 { // (100+200+300)/3, all tasks
		t.Errorf("mean tokens = %v, want 200", meanTok)
	}
}

func TestValidate_RejectsBadSets(t *testing.T) {
	cases := map[string]TaskSet{
		"no sha": {Tasks: []Task{{ID: "x", Query: "q", Bucket: Localization, Gold: []Gold{{Path: "a.py", Start: 1, End: 2}}}}},
		"dup id": {TargetSHA: "abc", Tasks: []Task{
			{ID: "x", Query: "q", Bucket: Localization, Gold: []Gold{{Path: "a.py", Start: 1, End: 1}}},
			{ID: "x", Query: "q", Bucket: Localization, Gold: []Gold{{Path: "a.py", Start: 1, End: 1}}},
		}},
		"bad bucket": {TargetSHA: "abc", Tasks: []Task{{ID: "x", Query: "q", Bucket: "weird", Gold: []Gold{{Path: "a.py", Start: 1, End: 1}}}}},
		"no gold":    {TargetSHA: "abc", Tasks: []Task{{ID: "x", Query: "q", Bucket: Localization}}},
		"bad span":   {TargetSHA: "abc", Tasks: []Task{{ID: "x", Query: "q", Bucket: Localization, Gold: []Gold{{Path: "a.py", Start: 5, End: 2}}}}},
	}
	for name, ts := range cases {
		if err := ts.Validate(); err == nil {
			t.Errorf("%s: expected validation error, got nil", name)
		}
	}
}

func TestValidate_AcceptsGoodSet(t *testing.T) {
	ts := TaskSet{
		TargetSHA: "1651351386ab31d8ae492c8a4922797714ca97d1",
		Tasks: []Task{
			{ID: "x", Query: "how does X work", Bucket: Localization, Gold: []Gold{{Path: "a.py", Start: 1, End: 10}}},
		},
	}
	if err := ts.Validate(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestLoad_FrozenSet(t *testing.T) {
	set, err := Load("tasks.yaml")
	if err != nil {
		t.Fatalf("load frozen set: %v", err)
	}
	if set.TargetSHA != "1651351386ab31d8ae492c8a4922797714ca97d1" {
		t.Errorf("unexpected pinned SHA %q", set.TargetSHA)
	}
	if len(set.Tasks) < 10 {
		t.Errorf("frozen set has only %d tasks", len(set.Tasks))
	}
	var loc int
	for _, task := range set.Tasks {
		if task.Bucket == Localization {
			loc++
		}
	}
	if loc == 0 {
		t.Error("frozen set has no localization tasks (the headline would be empty)")
	}
}

func TestRenderMarkdown(t *testing.T) {
	set := &TaskSet{TargetRepo: "github.com/django/django", TargetSHA: "abc123"}
	results := []TaskResult{
		{ID: "loc1", Bucket: Localization, GoldSize: 2, Recall: 1.0, RetrievedTokens: 120},
		{ID: "enum1", Bucket: Enumeration, GoldSize: 40, Recall: 0.2, RetrievedTokens: 350},
	}
	md := RenderMarkdown(set, results, 10, "- embedder: `ollama` / `mxbai-embed-large`")
	for _, want := range []string{"recall@10", "abc123", "loc1", "enum1", "mean retrieved_code tokens", "mxbai-embed-large"} {
		if !strings.Contains(md, want) {
			t.Errorf("rendered report missing %q\n%s", want, md)
		}
	}
}
