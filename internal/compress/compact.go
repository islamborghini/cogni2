package compress

import (
	"context"
	"fmt"
)

// CompactingCompressor is the cache-safe Stage 3 policy. Where GuidelineCompressor
// is called every turn and rewrites the oldest over-budget observation each time it
// overflows — which moves the prompt's first-divergence point forward turn after
// turn and busts a frontier API's prefix cache repeatedly (measured in
// bench/results/stage3-cache.md: -13% to -17% net cost at a frontier read discount)
// — this compactor compacts on HYSTERESIS. It stays a no-op until history crosses
// the high-water mark (budget), then summarizes the whole eligible backlog in ONE
// pass down to a low-water target (TargetFraction*budget) and freezes it. That makes
// at most one cache divergence per checkpoint, with headroom before the next, so
// append-only growth re-caches against the new, smaller prefix instead of re-paying
// full price every turn.
//
// It shares GuidelineCompressor's invariants and helpers: the goal and the
// most-recent action+observation are never touched (protectedMask), each observation
// is summarized at most once (the Compressed watermark), and observations are
// summarized rather than deleted, keeping the path:line anchor for re-reads. The only
// difference is WHEN and HOW MUCH it compacts — rarely, and well below budget.
type CompactingCompressor struct {
	Summarizer Summarizer
	Tok        Tokenizer
	Guideline  string
	// TargetFraction is the low-water mark as a fraction of budget: a checkpoint
	// compacts down to TargetFraction*budget (not merely under budget) so the next
	// checkpoint is far off. Rare compaction is what keeps the cache stable. A value
	// outside (0,1) defaults to 0.5.
	TargetFraction float64
	// MinSummarizeTokens leaves observations at or below this size untouched —
	// summarizing a tiny output costs more (a call, more tokens) than it saves. 0
	// summarizes every eligible observation.
	MinSummarizeTokens int
}

var _ Compressor = (*CompactingCompressor)(nil)

// defaultTargetFraction is the low-water mark when TargetFraction is unset: compact
// down to half the budget, leaving headroom for several more turns before the next
// checkpoint.
const defaultTargetFraction = 0.5

// Compress runs at most one checkpoint. Below the high-water mark (budget) it is a
// genuine no-op: no summarizer calls, no edits, so the prefix never diverges and
// short tasks pay nothing. Once over budget it summarizes the oldest not-yet-
// compressed observations in a single pass until the history is at or below the
// low-water target, then drops the oldest if it still overflows. Because it compacts
// to the target (not just under budget), an immediate re-call is back under budget
// and returns the no-op — the idempotence the per-turn rewrite policy lacked.
func (c *CompactingCompressor) Compress(ctx context.Context, history []Turn, budget int) (Result, error) {
	out := make([]Turn, len(history))
	copy(out, history)
	res := Result{History: out}
	if len(out) == 0 || c.Tok == nil || budget <= 0 || totalTokens(out, c.Tok) <= budget {
		return res, nil
	}

	frac := c.TargetFraction
	if frac <= 0 || frac >= 1 {
		frac = defaultTargetFraction
	}
	target := int(float64(budget) * frac)

	protected := protectedMask(out)

	// The checkpoint: summarize the oldest not-yet-compressed observations until the
	// history is at or below the low-water target. All edits happen here, in one pass,
	// so the cache diverges at most once.
	for totalTokens(out, c.Tok) > target {
		i := oldestSummarizable(out, protected, c.Tok, c.MinSummarizeTokens)
		if i < 0 {
			break
		}
		summary, err := c.Summarizer.Summarize(ctx, out[i].Content, c.Guideline)
		if err != nil {
			return Result{}, fmt.Errorf("compact: summarize step %d: %w", out[i].Step, err)
		}
		out[i].Compressed = true
		if c.Tok.Count(summary) < c.Tok.Count(out[i].Content) {
			out[i].Content = summary
			res.Summarized++
		}
	}

	// If summarizing everything still overflows the budget, drop the oldest with a
	// visible marker — the protected region is never dropped, so the budget is advisory.
	for totalTokens(out, c.Tok) > budget {
		idx := oldestDroppable(out, protected)
		if idx < 0 {
			break
		}
		out[idx].Content = dropMarker
		res.Dropped++
	}
	return res, nil
}
