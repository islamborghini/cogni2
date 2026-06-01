package chunk

import (
	"sort"
	"strings"

	"github.com/islamborghini/cogni2/internal/retrieve"
	ts "github.com/tree-sitter/go-tree-sitter"
)

// rawChunk is the internal form carried through walking and merging. Byte
// offsets keep slices verbatim and let merges union ranges cheaply; content is
// set only for synthesized chunks (class headers) whose text is not a single
// contiguous source slice.
type rawChunk struct {
	kind      string
	startByte uint
	endByte   uint
	content   string // non-empty only when synthesized
}

type chunker struct {
	path       string
	src        []byte
	tok        Tokenizer
	max        int
	lineStarts []int
	out        []rawChunk
}

// walk visits the named children of a scope (module or class body), emitting a
// chunk per definition. inClass flips function → method.
func (c *chunker) walk(scope *ts.Node, inClass bool) {
	for i := uint(0); i < scope.NamedChildCount(); i++ {
		n := scope.NamedChild(i)
		switch n.Kind() {
		case "function_definition":
			c.emitFunc(n, n, inClass)
		case "class_definition":
			c.emitClass(n, n)
		case "decorated_definition":
			def := n.ChildByFieldName("definition")
			if def == nil {
				continue
			}
			switch def.Kind() {
			case "function_definition":
				c.emitFunc(n, def, inClass)
			case "class_definition":
				c.emitClass(n, def)
			}
		}
	}
	// Module-level imports, constants, and other statements are intentionally
	// not indexed: the cAST ablation shows tiny non-def chunks bloat the index
	// and dilute retrieval, and the LOCKED kind set has no slot for them.
}

// emitFunc emits a function/method chunk. outer carries the span and verbatim
// text (decorators included); def is the underlying function_definition used to
// find the body when an oversize unit must be split.
func (c *chunker) emitFunc(outer, def *ts.Node, inClass bool) {
	kind := retrieve.KindFunction
	if inClass {
		kind = retrieve.KindMethod
	}
	if c.fits(outer.StartByte(), outer.EndByte()) {
		c.emitBytes(outer.StartByte(), outer.EndByte(), kind)
		return
	}
	c.splitFunc(outer, def, kind)
}

// splitFunc handles a function/method larger than the budget: it emits the
// signature as one chunk, then splits the body along statement boundaries.
func (c *chunker) splitFunc(outer, def *ts.Node, kind string) {
	body := def.ChildByFieldName("body")
	if body == nil {
		c.emitLineSplit(outer.StartByte(), outer.EndByte(), kind)
		return
	}
	c.emitBytes(outer.StartByte(), body.StartByte(), kind) // signature (+ decorators)
	c.splitBlock(body, kind)
}

// splitBlock greedily groups a block's statements into chunks under the budget,
// descending to line-splitting only for a single statement that itself exceeds
// the budget.
func (c *chunker) splitBlock(body *ts.Node, kind string) {
	var start, end uint
	have := false
	flush := func() {
		if have {
			c.emitBytes(start, end, kind)
			have = false
		}
	}
	for i := uint(0); i < body.NamedChildCount(); i++ {
		st := body.NamedChild(i)
		s, e := st.StartByte(), st.EndByte()
		if !c.fits(s, e) {
			flush()
			c.emitLineSplit(s, e, kind)
			continue
		}
		switch {
		case !have:
			start, end, have = s, e, true
		case c.fits(start, e):
			end = e
		default:
			flush()
			start, end, have = s, e, true
		}
	}
	flush()
}

// emitLineSplit is the leaf fallback: it slices [s,e) into consecutive
// line-aligned windows that each fit the budget.
func (c *chunker) emitLineSplit(s, e uint, kind string) {
	startIdx := c.lineIndex(s)
	endIdx := c.lineIndex(e - 1)
	winStart := uint(c.lineStarts[startIdx])
	for li := startIdx; li <= endIdx; li++ {
		lineStart := uint(c.lineStarts[li])
		lineEnd := e
		if li+1 < len(c.lineStarts) && uint(c.lineStarts[li+1]) < e {
			lineEnd = uint(c.lineStarts[li+1])
		}
		// If appending this line overflows a non-empty window, flush first.
		if winStart < lineStart && !c.fits(winStart, lineEnd) {
			c.emitBytes(winStart, lineStart, kind)
			winStart = lineStart
		}
	}
	if winStart < e {
		c.emitBytes(winStart, e, kind)
	}
}

// emitClass emits a signatures-only class_header chunk plus a chunk per method
// body. outer carries the full class span (decorators included); def is the
// class_definition whose body is walked.
func (c *chunker) emitClass(outer, def *ts.Node) {
	body := def.ChildByFieldName("body")

	var b strings.Builder
	// Declaration: decorators + `class X(Base):`, trimmed of the trailing
	// newline/indent that precedes the first body statement.
	declEnd := def.EndByte()
	if body != nil {
		declEnd = body.StartByte()
	}
	b.WriteString(strings.TrimRight(c.text(outer.StartByte(), declEnd), " \t\n"))

	if body != nil {
		for i := uint(0); i < body.NamedChildCount(); i++ {
			st := body.NamedChild(i)
			switch st.Kind() {
			case "expression_statement":
				if inner := st.NamedChild(0); inner != nil {
					switch inner.Kind() {
					case "string": // class docstring
						b.WriteString("\n    " + c.collapse(st.StartByte(), st.EndByte()))
					case "assignment": // class-level attribute
						b.WriteString("\n    " + c.text(st.StartByte(), st.EndByte()))
					}
				}
			case "function_definition":
				b.WriteString("\n    " + c.signature(st, st))
				c.emitFunc(st, st, true)
			case "decorated_definition":
				d := st.ChildByFieldName("definition")
				if d == nil {
					continue
				}
				switch d.Kind() {
				case "function_definition":
					b.WriteString("\n    " + c.signature(st, d))
					c.emitFunc(st, d, true)
				case "class_definition":
					b.WriteString("\n    " + c.signature(st, d))
					c.emitClass(st, d)
				}
			case "class_definition":
				b.WriteString("\n    " + c.signature(st, st))
				c.emitClass(st, st)
			}
		}
	}

	c.out = append(c.out, rawChunk{
		kind:      retrieve.KindClassHeader,
		startByte: outer.StartByte(),
		endByte:   outer.EndByte(),
		content:   c.capToBudget(b.String()),
	})
}

// capToBudget trims a synthesized chunk to the token budget at a line boundary.
// A class header for a class with very many methods can otherwise exceed both
// the budget and an embedder's context window; the dropped signatures are not
// lost to retrieval because each method also has its own chunk. The class
// declaration line is always kept.
func (c *chunker) capToBudget(s string) string {
	if c.tok.Count(s) <= c.max {
		return s
	}
	lines := strings.Split(s, "\n")
	var b strings.Builder
	b.WriteString(lines[0])
	for _, ln := range lines[1:] {
		if c.tok.Count(b.String()+"\n"+ln) > c.max {
			break
		}
		b.WriteByte('\n')
		b.WriteString(ln)
	}
	return b.String()
}

// signature returns the declaration line(s) of def (decorators included via
// outer) up to its body, with the trailing colon/whitespace replaced by " ...".
func (c *chunker) signature(outer, def *ts.Node) string {
	end := def.EndByte()
	if body := def.ChildByFieldName("body"); body != nil {
		end = body.StartByte()
	}
	sig := strings.TrimRight(c.text(outer.StartByte(), end), " \t\n:")
	return sig + " ..."
}

// mergeSameKind greedily packs adjacent same-kind verbatim chunks into one chunk
// while the union stays within budget. Synthesized chunks (class headers) are
// never merged — joining their byte ranges would re-include method bodies.
func (c *chunker) mergeSameKind() {
	if len(c.out) < 2 {
		return
	}
	merged := c.out[:1]
	for _, cur := range c.out[1:] {
		prev := &merged[len(merged)-1]
		if cur.content == "" && prev.content == "" &&
			cur.kind == prev.kind &&
			c.fits(prev.startByte, cur.endByte) {
			prev.endByte = cur.endByte
			continue
		}
		merged = append(merged, cur)
	}
	c.out = merged
}

// project converts internal chunks to the public form, deriving verbatim content
// from byte ranges and mapping offsets to 1-based inclusive line numbers.
func (c *chunker) project() []Chunk {
	out := make([]Chunk, 0, len(c.out))
	for _, r := range c.out {
		content := r.content
		if content == "" {
			content = c.text(r.startByte, r.endByte)
		}
		out = append(out, Chunk{
			Path:      c.path,
			StartLine: c.lineIndex(r.startByte) + 1,
			EndLine:   c.lineIndex(r.endByte-1) + 1,
			Kind:      r.kind,
			Content:   content,
		})
	}
	return out
}

func (c *chunker) text(s, e uint) string { return string(c.src[s:e]) }

func (c *chunker) emitBytes(s, e uint, kind string) {
	c.out = append(c.out, rawChunk{kind: kind, startByte: s, endByte: e})
}

func (c *chunker) fits(s, e uint) bool {
	return c.tok.Count(c.text(s, e)) <= c.max
}

// collapse returns a node's text with internal newlines flattened to single
// spaces, used to keep a multi-line docstring on one skeleton line.
func (c *chunker) collapse(s, e uint) string {
	return strings.Join(strings.Fields(c.text(s, e)), " ")
}

// lineIndex returns the 0-based line index containing byte offset b.
func (c *chunker) lineIndex(b uint) int {
	idx := sort.Search(len(c.lineStarts), func(i int) bool {
		return uint(c.lineStarts[i]) > b
	})
	return idx - 1
}
