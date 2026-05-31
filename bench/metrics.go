package bench

import (
	"fmt"
	"sort"
	"strings"

	"github.com/islamborghini/cogni2/internal/retrieve"
)

// TaskResult is one task's scored outcome.
type TaskResult struct {
	ID              string
	Bucket          Bucket
	GoldSize        int
	Recall          float64
	RetrievedTokens int
}

// goldHit reports whether any retrieved chunk overlaps the gold span: same file
// and intersecting line ranges.
func goldHit(g Gold, retrieved []retrieve.RetrievedChunk) bool {
	for _, c := range retrieved {
		if c.Path == g.Path && c.StartLine <= g.End && g.Start <= c.EndLine {
			return true
		}
	}
	return false
}

// Recall is the fraction of a task's gold spans whose containing chunk appears
// in the retrieved set.
func Recall(gold []Gold, retrieved []retrieve.RetrievedChunk) float64 {
	if len(gold) == 0 {
		return 0
	}
	hits := 0
	for _, g := range gold {
		if goldHit(g, retrieved) {
			hits++
		}
	}
	return float64(hits) / float64(len(gold))
}

// Headline reduces per-task results to the two Stage 1 numbers. recall@10 is
// macro-averaged over localization tasks only (the paper-comparable figure);
// mean retrieved tokens is averaged over every task.
func Headline(results []TaskResult) (recallAt10, meanRetrievedTokens float64) {
	var locSum float64
	var locN int
	var tokSum, tokN int
	for _, r := range results {
		if r.Bucket == Localization {
			locSum += r.Recall
			locN++
		}
		tokSum += r.RetrievedTokens
		tokN++
	}
	if locN > 0 {
		recallAt10 = locSum / float64(locN)
	}
	if tokN > 0 {
		meanRetrievedTokens = float64(tokSum) / float64(tokN)
	}
	return recallAt10, meanRetrievedTokens
}

// RenderMarkdown produces bench/results/stage1.md: the headline numbers, a
// per-task table, and the honesty note on enumeration tasks.
func RenderMarkdown(set *TaskSet, results []TaskResult, k int) string {
	recall, meanTok := Headline(results)

	sorted := append([]TaskResult(nil), results...)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].Bucket != sorted[j].Bucket {
			return sorted[i].Bucket == Localization // localization first
		}
		return sorted[i].ID < sorted[j].ID
	})

	var b strings.Builder
	fmt.Fprintf(&b, "# Stage 1 — cAST retrieval baseline\n\n")
	fmt.Fprintf(&b, "Target: `%s` @ `%s`\n\n", set.TargetRepo, set.TargetSHA)
	fmt.Fprintf(&b, "## Headline\n\n")
	fmt.Fprintf(&b, "- **recall@%d (localization, macro-avg): %.3f**\n", k, recall)
	fmt.Fprintf(&b, "- **mean retrieved_code tokens: %.0f**\n\n", meanTok)
	fmt.Fprintf(&b, "recall@%d is averaged over localization tasks only. Enumeration tasks "+
		"(|gold| > %d) are bounded by %d/|gold| by construction and are listed for reference, "+
		"not folded into the headline.\n\n", k, k, k)

	fmt.Fprintf(&b, "## Per task\n\n")
	fmt.Fprintf(&b, "| task | bucket | gold | recall@%d | retrieved_tokens |\n", k)
	fmt.Fprintf(&b, "|---|---|---:|---:|---:|\n")
	for _, r := range sorted {
		fmt.Fprintf(&b, "| %s | %s | %d | %.3f | %d |\n",
			r.ID, r.Bucket, r.GoldSize, r.Recall, r.RetrievedTokens)
	}
	b.WriteString("\n")
	return b.String()
}
