// Package meter records, per task run, where an agent's tokens actually go.
//
// This is the instrument the entire project rests on: the headline claim ("same
// success rate, N% fewer tokens") is only credible if token counts are real
// (tiktoken-compatible, not a word count) and bucketed by source so each stage's
// reduction can be attributed to the component that produced it.
package meter

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// Token buckets. Every counted string is attributed to exactly one bucket.
const (
	BucketRetrievedCode = "retrieved_code" // code assembled into context by retrieval
	BucketHistory       = "history"        // accumulated prior turns / observations
	BucketSystem        = "system"         // system prompt + tool definitions
	BucketOutput        = "output"         // model output for the task
)

// Buckets is the canonical ordered set, used for stable report columns.
var Buckets = []string{BucketRetrievedCode, BucketHistory, BucketSystem, BucketOutput}

// Tokenizer counts tokens the way the target model's API bills them. Behind an
// interface so the real tiktoken-compatible implementation is swappable (and so
// tests can inject a deterministic counter).
type Tokenizer interface {
	Count(text string) int
}

// Record is the persisted per-task token accounting for one stage run.
type Record struct {
	TaskID  string         `json:"task_id"`
	Stage   int            `json:"stage"`
	Buckets map[string]int `json:"buckets"`
	Total   int            `json:"total"`
}

// Meter accumulates token counts for a single task run.
type Meter struct {
	tok     Tokenizer
	taskID  string
	stage   int
	buckets map[string]int
}

// New starts metering one task run at a given stage.
func New(tok Tokenizer, stage int, taskID string) *Meter {
	return &Meter{tok: tok, taskID: taskID, stage: stage, buckets: map[string]int{}}
}

// Add tokenizes text and credits it to bucket. Unknown buckets are allowed but
// callers should prefer the Bucket* constants so reports line up across stages.
func (m *Meter) Add(bucket, text string) {
	m.buckets[bucket] += m.tok.Count(text)
}

// AddTokens credits a pre-counted amount to bucket (for sources already measured
// upstream, e.g. an API usage field).
func (m *Meter) AddTokens(bucket string, n int) {
	m.buckets[bucket] += n
}

// Record snapshots the current counts into a persistable record.
func (m *Meter) Record() Record {
	b := make(map[string]int, len(m.buckets))
	total := 0
	for k, v := range m.buckets {
		b[k] = v
		total += v
	}
	return Record{TaskID: m.taskID, Stage: m.stage, Buckets: b, Total: total}
}

// Persist writes the record to bench/runs/stage<N>/<task_id>.json under root.
func (m *Meter) Persist(root string) (string, error) {
	rec := m.Record()
	dir := filepath.Join(root, "bench", "runs", fmt.Sprintf("stage%d", rec.Stage))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, rec.TaskID+".json")
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", err
	}
	return path, nil
}

// MeanByBucket aggregates a set of records into mean tokens per bucket and the
// mean total. Used by the stage report generators.
func MeanByBucket(recs []Record) (perBucket map[string]float64, meanTotal float64) {
	perBucket = map[string]float64{}
	if len(recs) == 0 {
		return perBucket, 0
	}
	sum := map[string]int{}
	totalSum := 0
	for _, r := range recs {
		for k, v := range r.Buckets {
			sum[k] += v
		}
		totalSum += r.Total
	}
	n := float64(len(recs))
	for k, v := range sum {
		perBucket[k] = float64(v) / n
	}
	return perBucket, float64(totalSum) / n
}

// SortedBuckets returns bucket names present in a record in canonical order,
// with any extras appended alphabetically.
func SortedBuckets(b map[string]int) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(b))
	for _, k := range Buckets {
		if _, ok := b[k]; ok {
			out = append(out, k)
			seen[k] = true
		}
	}
	var extra []string
	for k := range b {
		if !seen[k] {
			extra = append(extra, k)
		}
	}
	sort.Strings(extra)
	return append(out, extra...)
}
