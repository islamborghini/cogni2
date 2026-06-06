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
	// Compressed marks an observation that has already been summarized in a prior
	// step. The agent carries this back across turns so each observation is
	// summarized at most once — compression is incremental, not re-run over the
	// whole history every step (which is O(turns²) summarizer cost).
	Compressed bool
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

// Compress condenses history to fit budget tokens. It only acts when the history
// exceeds the budget (nothing to save otherwise), summarizes the oldest not-yet-
// compressed observations one at a time until it fits, then drops the oldest if it
// still doesn't. The goal (first user turn) and the most recent action+observation
// are never touched. A no-op (budget <= 0, no tokenizer, or already under budget)
// makes no summarizer calls — the key fix that keeps overhead from dwarfing the
// savings.
func (c *GuidelineCompressor) Compress(ctx context.Context, history []Turn, budget int) (Result, error) {
	out := make([]Turn, len(history))
	copy(out, history)
	res := Result{History: out}
	if len(out) == 0 || c.Tok == nil || budget <= 0 || c.total(out) <= budget {
		return res, nil
	}

	protected := protectedMask(out)

	// Summarize the oldest not-yet-compressed observations until the history fits.
	// Marking each Compressed (even if the summary didn't shrink it) means it is
	// never summarized again on a later step.
	for c.total(out) > budget {
		i := oldestSummarizable(out, protected, c.Tok, c.MinSummarizeTokens)
		if i < 0 {
			break
		}
		summary, err := c.Summarizer.Summarize(ctx, out[i].Content, c.Guideline)
		if err != nil {
			return Result{}, fmt.Errorf("compress: summarize step %d: %w", out[i].Step, err)
		}
		out[i].Compressed = true
		if c.Tok.Count(summary) < c.Tok.Count(out[i].Content) {
			out[i].Content = summary
			res.Summarized++
		}
	}

	// If summarizing everything still didn't fit, drop the oldest observations.
	// The protected region is never dropped, so the budget stays advisory.
	for c.total(out) > budget {
		idx := oldestDroppable(out, protected)
		if idx < 0 {
			break
		}
		out[idx].Content = dropMarker
		res.Dropped++
	}
	return res, nil
}

// oldestSummarizable returns the earliest observation eligible to be summarized:
// a non-protected tool turn, not already compressed, not already the drop marker,
// and (if MinSummarizeTokens is set) larger than that floor.
func oldestSummarizable(turns []Turn, protected []bool, tok Tokenizer, minTokens int) int {
	for i := range turns {
		if protected[i] || turns[i].Role != RoleTool || turns[i].Compressed || turns[i].Content == dropMarker {
			continue
		}
		if minTokens > 0 && tok != nil && tok.Count(turns[i].Content) <= minTokens {
			continue
		}
		return i
	}
	return -1
}

// total is the token count of the whole history under the configured tokenizer.
func (c *GuidelineCompressor) total(turns []Turn) int {
	return totalTokens(turns, c.Tok)
}

// totalTokens sums the content tokens of a history under tok. Shared by both the
// GuidelineCompressor and the CompactingCompressor so the two policies measure the
// budget identically.
func totalTokens(turns []Turn, tok Tokenizer) int {
	n := 0
	for _, t := range turns {
		n += tok.Count(t.Content)
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
