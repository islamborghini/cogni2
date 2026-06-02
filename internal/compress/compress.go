// Package compress implements Stage 3 ACON-style history compression: between
// agent steps it condenses older observations (prior tool outputs, file reads)
// into concise summaries while ALWAYS keeping the goal and the most recent
// action+observation pair verbatim, so fresh state is never lost. Summaries are
// produced by a Summarizer model under an editable natural-language guideline
// (guideline.txt) that names what must be preserved — paths, identifiers,
// errors, decisions — and what may be dropped.
//
// Like Stage 2 (internal/skeleton), the protected region is never mutated; older
// observations are summarized rather than silently deleted, and observations that
// still overflow the budget are dropped oldest-first with a visible marker and
// counted. Recall-style safety is not this package's job — it carries no LLM in
// the hot path beyond the Summarizer and is unit-tested entirely offline with a
// FakeSummarizer.
package compress

import (
	"context"
	"fmt"
	"strings"
)

// Turn roles, matching the OpenAI chat-completions message roles the agent loop
// produces. The compressor treats "tool" turns as observations — the bulky,
// compressible content — and leaves the goal and the most recent action verbatim.
const (
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Turn is one message in the agent's accumulated history. It is the type the
// compressor contract is written against; the agent loop reuses it verbatim.
// Kind is an origin tag the agent sets for token-bucket attribution (e.g. marking
// turns that carry retrieved code); the compressor keys its decisions off Role.
type Turn struct {
	Role    string
	Content string
	Step    int
	Kind    string
}

// Summarizer condenses text under a natural-language instruction. It is the one
// LLM dependency of the compressor, behind an interface so a cheap model (Groq
// llama-3.1-8b by default) or a deterministic Fake can be injected.
type Summarizer interface {
	Summarize(ctx context.Context, text, instruction string) (string, error)
}

// Tokenizer counts tokens against the history budget. Restated locally (like
// internal/skeleton and internal/chunk) so this package does not import meter.
type Tokenizer interface {
	Count(text string) int
}

// Result is the outcome of one compression pass.
type Result struct {
	History    []Turn // the compressed history, same order and length as the input
	Summarized int    // observations replaced by a (shorter) summary
	Dropped    int    // observations dropped entirely to fit the budget, replaced by dropMarker
}

// Compressor condenses agent history between steps. The agent loop holds one
// (nil for the uncompressed baseline arm).
type Compressor interface {
	Compress(ctx context.Context, history []Turn, budget int) (Result, error)
}

// dropMarker replaces an observation evicted to fit the budget. Kept short so a
// drop always reduces tokens, but visible so the agent and the transcript can see
// that earlier context was elided.
const dropMarker = "[earlier observation omitted]"

// GuidelineCompressor is the ACON-style compressor: keep the goal and the most
// recent action+observation verbatim, summarize older observations under the
// guideline, then drop the oldest summaries if the history is still over budget.
type GuidelineCompressor struct {
	Summarizer Summarizer
	Tok        Tokenizer
	Guideline  string
	// MinSummarizeTokens leaves observations already at or below this size
	// untouched — summarizing a tiny output costs more tokens (and an API call)
	// than it saves. 0 means summarize every older observation.
	MinSummarizeTokens int
}

var _ Compressor = (*GuidelineCompressor)(nil)

// Compress condenses history to roughly budget tokens. It never mutates the goal
// (the first user turn) or the most recent action+observation (the last assistant
// turn through the end). budget <= 0 disables the drop pass (summarize only).
func (c *GuidelineCompressor) Compress(ctx context.Context, history []Turn, budget int) (Result, error) {
	out := make([]Turn, len(history))
	copy(out, history)
	res := Result{History: out}
	if len(out) == 0 {
		return res, nil
	}

	protected := protectedMask(out)

	// Pass 1 — summarize older observations under the guideline.
	for i := range out {
		if protected[i] || out[i].Role != RoleTool {
			continue
		}
		if c.Tok != nil && c.MinSummarizeTokens > 0 && c.Tok.Count(out[i].Content) <= c.MinSummarizeTokens {
			continue
		}
		summary, err := c.Summarizer.Summarize(ctx, out[i].Content, c.Guideline)
		if err != nil {
			return Result{}, fmt.Errorf("compress: summarize step %d: %w", out[i].Step, err)
		}
		// Only accept a summary that actually shrinks the observation; a summarizer
		// that expands the text helps nobody.
		if c.Tok == nil || c.Tok.Count(summary) < c.Tok.Count(out[i].Content) {
			out[i].Content = summary
			res.Summarized++
		}
	}

	// Pass 2 — if still over budget, drop the oldest summarized observations first.
	// The protected region (goal + most recent pair) is never dropped, so the budget
	// is advisory: if it alone exceeds the budget we stop rather than break it.
	for c.Tok != nil && budget > 0 && c.total(out) > budget {
		idx := oldestDroppable(out, protected)
		if idx < 0 {
			break
		}
		out[idx].Content = dropMarker
		res.Dropped++
	}
	return res, nil
}

// total is the token count of the whole history under the configured tokenizer.
func (c *GuidelineCompressor) total(turns []Turn) int {
	n := 0
	for _, t := range turns {
		n += c.Tok.Count(t.Content)
	}
	return n
}

// protectedMask marks the turns the compressor must keep verbatim: the goal (the
// first user turn) and the most recent action+observation (the last assistant turn
// and everything after it — its tool result(s)). This is the ACON rule that keeps
// fresh state and the task objective intact.
func protectedMask(turns []Turn) []bool {
	p := make([]bool, len(turns))
	for i := range turns {
		if turns[i].Role == RoleUser {
			p[i] = true // the goal
			break
		}
	}
	lastAction := -1
	for i := range turns {
		if turns[i].Role == RoleAssistant {
			lastAction = i
		}
	}
	if lastAction >= 0 {
		for i := lastAction; i < len(turns); i++ {
			p[i] = true
		}
	}
	return p
}

// oldestDroppable returns the index of the earliest observation that may be
// dropped — a non-protected tool turn not already replaced by a marker — or -1
// when none remain.
func oldestDroppable(turns []Turn, protected []bool) int {
	for i := range turns {
		if protected[i] || turns[i].Role != RoleTool || turns[i].Content == dropMarker {
			continue
		}
		return i
	}
	return -1
}

// FakeSummarizer is a deterministic, offline Summarizer for hermetic tests: it
// keeps the first line of the text (which carries the path:line re-Read anchor)
// and appends a marker, discarding the rest. The same text always yields the same
// summary, and any multi-line input shrinks — so tests need no network or API key.
type FakeSummarizer struct{}

// Summarize implements Summarizer.
func (FakeSummarizer) Summarize(_ context.Context, text, _ string) (string, error) {
	first := text
	if i := strings.IndexByte(text, '\n'); i >= 0 {
		first = text[:i]
	}
	return first + " …[summarized]", nil
}
