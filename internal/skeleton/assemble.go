package skeleton

import (
	"strings"

	"github.com/islamborghini/cogni2/internal/retrieve"
)

// FullBodyBudgetFraction is the share of the assembly budget reserved for full
// bodies of the top-ranked chunks before the rest are rendered as skeletons.
//
// Fixed at 0.6 for Stage 2 by design: Stage 2 has no quality metric that responds
// to it (recall is flat, syntax is always valid), so varying it here only moves
// the token count with nothing to justify a setting. It is swept against
// end-to-end success rate at Stage 3, the first metric that can see what it trades.
const FullBodyBudgetFraction = 0.6

// blockSep separates rendered chunks in the assembled context.
const blockSep = "\n\n"

// Tokenizer counts tokens against the budget. Restated locally (like
// chunk.Tokenizer) so this package does not import meter.
type Tokenizer interface {
	Count(text string) int
}

// Assembly is the result of packing retrieved chunks into a token budget.
type Assembly struct {
	Text           string // the assembled context
	FullCount      int    // chunks rendered as full bodies
	SkeletonCount  int    // chunks rendered as skeletons (kept, with a re-Read anchor)
	Dropped        int    // chunks evicted entirely — NO representation, not even an anchor
	KeptFullTokens int    // tokens the KEPT chunks would cost as full bodies
}

// Assemble packs chunks (already Score-sorted, highest first) into budget tokens.
// The top chunks are rendered as full bodies until ~FullBodyBudgetFraction of the
// budget is consumed; the remainder are rendered as skeletons; if the total still
// exceeds budget, the lowest-scored skeletons are dropped last. Every emitted
// chunk carries a visible path:start-end header.
//
// Dropped counts chunks left with NO representation (not even a skeleton anchor):
// recall@10 overstates visible content wherever Dropped > 0, which is why Dropped,
// not recall, is half the Stage 2 safety claim. KeptFullTokens is the denominator
// for the skeletonization-isolated reduction — what the same kept chunks would
// have cost as full bodies, holding the chunk set fixed.
func Assemble(chunks []retrieve.RetrievedChunk, budget int, tok Tokenizer) (Assembly, error) {
	var a Assembly
	if len(chunks) == 0 {
		return a, nil
	}

	fullBudget := int(float64(budget) * FullBodyBudgetFraction)
	var fulls, skels []string

	i := 0
	for ; i < len(chunks); i++ {
		blk := chunks[i].Header() + "\n" + chunks[i].Content
		// Always include the top-ranked chunk full; otherwise stop before the one
		// that would cross the full-body share and skeletonize the rest.
		if i > 0 && tok.Count(join(fulls)+blockSep+blk) > fullBudget {
			break
		}
		fulls = append(fulls, blk)
	}
	for ; i < len(chunks); i++ {
		s, err := Skeleton(chunks[i])
		if err != nil {
			return Assembly{}, err
		}
		skels = append(skels, chunks[i].Header()+"\n"+s)
	}

	// Evict lowest-scored skeletons (tail of the rank order) until within budget.
	// Full bodies are never dropped — they are the highest-value chunks.
	for len(skels) > 0 && tok.Count(join(combine(fulls, skels))) > budget {
		skels = skels[:len(skels)-1]
		a.Dropped++
	}

	a.FullCount = len(fulls)
	a.SkeletonCount = len(skels)
	a.Text = join(combine(fulls, skels))

	// The dropped chunks are exactly the lowest-scored tail, so the kept set is the
	// prefix chunks[:len-Dropped]. Cost it as full bodies for the (a) comparator.
	kept := chunks[:len(chunks)-a.Dropped]
	keptFull := make([]string, 0, len(kept))
	for _, c := range kept {
		keptFull = append(keptFull, c.Header()+"\n"+c.Content)
	}
	a.KeptFullTokens = tok.Count(join(keptFull))

	return a, nil
}

func join(blocks []string) string { return strings.Join(blocks, blockSep) }

func combine(a, b []string) []string {
	out := make([]string, 0, len(a)+len(b))
	out = append(out, a...)
	return append(out, b...)
}
