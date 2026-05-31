// Package chunk implements cAST: structural chunking of source via its AST
// (Zhang et al., arXiv 2506.15655). It walks the tree-sitter tree exposed by
// internal/parse and emits one chunk per syntactically complete unit — function
// body, method body, and a signatures-only class header — splitting any unit
// that exceeds the token budget along the next AST level down (line-based as a
// last resort) and greedily merging small adjacent same-kind siblings.
//
// Chunk kinds are the LOCKED retrieve.Kind* values; class_header carries method
// signatures only (no bodies), which intentionally diverges from cAST's
// verbatim-reproduction property to feed the Stage 2 skeleton compressor.
package chunk

import (
	"github.com/islamborghini/cogni2/internal/parse"
)

// DefaultMaxChunkTokens caps a chunk's size. The cAST paper budgets 2000
// non-whitespace characters; we denominate in tokens instead because the whole
// project is token-accounted and the meter is the source of truth. 800 tokens
// is a comparable code budget.
const DefaultMaxChunkTokens = 800

// Chunk is one retrievable unit of source. StartLine/EndLine are 1-based and
// inclusive. Content is verbatim source for function/method chunks and a
// synthesized skeleton for class_header chunks.
type Chunk struct {
	Path      string
	StartLine int
	EndLine   int
	Kind      string
	Content   string
}

// Options tunes the chunker. The zero value is usable: MaxChunkTokens falls back
// to DefaultMaxChunkTokens and Merge defaults off, so callers should set Merge
// explicitly (the eval runs with Merge=true, faithful to the paper).
type Options struct {
	MaxChunkTokens int
	Merge          bool
}

// Tokenizer counts tokens for the size budget. meter.Tokenizer satisfies it; the
// interface is restated here so chunk does not import meter.
type Tokenizer interface {
	Count(text string) int
}

// File chunks one Python source file. path is the repo-relative path stamped on
// every emitted chunk.
func File(path string, src []byte, tok Tokenizer, opts Options) ([]Chunk, error) {
	if opts.MaxChunkTokens <= 0 {
		opts.MaxChunkTokens = DefaultMaxChunkTokens
	}
	tree, err := parse.PythonTree(src)
	if err != nil {
		return nil, err
	}
	defer tree.Close()

	c := &chunker{
		path:       path,
		src:        src,
		tok:        tok,
		max:        opts.MaxChunkTokens,
		lineStarts: computeLineStarts(src),
	}
	c.walk(tree.RootNode(), false)
	if opts.Merge {
		c.mergeSameKind()
	}
	return c.project(), nil
}

// computeLineStarts records the byte offset at which each line begins, so a byte
// offset can be mapped to a 1-based line number by binary search.
func computeLineStarts(src []byte) []int {
	starts := make([]int, 1, len(src)/24+1)
	for i, b := range src {
		if b == '\n' {
			starts = append(starts, i+1)
		}
	}
	return starts
}
