// Package skeleton implements Stage 2 skeleton-first compression: it renders a
// retrieved chunk as its signature(s) + first-paragraph docstring + a body
// placeholder instead of the full body, without ever mutating code. Bodies are
// OMITTED — replaced by "...  # body omitted — expand via path:start-end" — never
// pruned token by token, so every emitted skeleton is itself valid Python and the
// agent can re-Read the exact span on demand.
//
// Stage 2 changes only how retrieved chunks are rendered; retrieval (Stage 1) is
// untouched, so recall is unchanged by construction.
package skeleton

import (
	"fmt"
	"strings"

	"github.com/islamborghini/cogni2/internal/parse"
	"github.com/islamborghini/cogni2/internal/retrieve"
	ts "github.com/tree-sitter/go-tree-sitter"
)

// Skeleton renders one retrieved chunk in skeleton form. class_header chunks are
// already skeletal and returned unchanged. function/method chunks are parsed and
// each top-level definition has its body replaced by a placeholder, keeping the
// signature (decorators included) and first-paragraph docstring.
//
// Anything that does not parse cleanly into at least one definition with a body —
// the signature-only or body-only fragments the cAST chunker emits for oversized
// units — is returned verbatim. That is always safe (no mutation) and keeps the
// function total. err is reserved for a parser-infrastructure failure; unparseable
// content is not an error, it is a verbatim passthrough.
func Skeleton(c retrieve.RetrievedChunk) (string, error) {
	if c.Kind == retrieve.KindClassHeader {
		return c.Content, nil // already signatures-only
	}

	src := []byte(c.Content)
	tree, err := parse.PythonTree(src)
	if err != nil {
		return "", err
	}
	defer tree.Close()

	root := tree.RootNode()
	if root.HasError() {
		return c.Content, nil // fragment or malformed slice — never risk corrupting it
	}

	// Collect the top-level function definitions we can skeletonize. Classes and
	// loose statements are left verbatim; a function missing its body (a split
	// signature fragment) has nothing to omit.
	type target struct {
		outer *ts.Node // decorated_definition or function_definition (carries decorators + span)
		body  *ts.Node
	}
	var targets []target
	for i := uint(0); i < root.NamedChildCount(); i++ {
		n := root.NamedChild(i)
		def := n
		if n.Kind() == "decorated_definition" {
			d := n.ChildByFieldName("definition")
			if d == nil {
				continue
			}
			def = d
		}
		if def.Kind() != "function_definition" {
			continue
		}
		body := def.ChildByFieldName("body")
		if body == nil || body.NamedChildCount() == 0 {
			continue // no body to omit (e.g. a split signature fragment) — leave verbatim
		}
		targets = append(targets, target{outer: n, body: body})
	}
	if len(targets) == 0 {
		return c.Content, nil
	}

	var b strings.Builder
	cursor := uint(0)
	for _, t := range targets {
		b.Write(src[cursor:t.outer.StartByte()]) // verbatim gap: blank lines, comments, stray stmts
		writeSkeleton(&b, t.outer, t.body, src, c)
		cursor = t.outer.EndByte()
	}
	b.Write(src[cursor:])
	return b.String(), nil
}

// writeSkeleton emits one definition reduced to:
//
//	<decorators + signature>:
//	    <docstring?>
//	    ...  # body omitted — expand via path:start-end
func writeSkeleton(b *strings.Builder, outer, body *ts.Node, src []byte, c retrieve.RetrievedChunk) {
	// Signature: decorators (via outer) through the start of the body, trailing
	// colon/whitespace trimmed then ":" restored. Mirrors internal/chunk.signature.
	sig := strings.TrimRight(string(src[outer.StartByte():body.StartByte()]), " \t\n:")
	b.WriteString(sig)
	b.WriteString(":\n")

	indent := defLineIndent(outer, src) + "    "

	if doc := parse.FirstDocstring(body, src); doc != "" {
		b.WriteString(indent)
		b.WriteString(pyDocstring(doc, indent))
		b.WriteByte('\n')
	}

	// Per-def precise re-Read anchor: the chunk's Content begins at c.StartLine, so
	// a node at row r maps to file line c.StartLine+r. A merged multi-def chunk thus
	// yields one exact span per body, not the whole-chunk range.
	startLn := c.StartLine + int(outer.StartPosition().Row)
	endLn := c.StartLine + int(outer.EndPosition().Row)
	fmt.Fprintf(b, "%s...  # body omitted — expand via %s:%d-%d", indent, c.Path, startLn, endLn)
}

// defLineIndent returns the leading whitespace of the line on which node n
// starts, or "" if n does not begin a line (the body placeholder is then indented
// one level past it). Keeps a method skeleton's body indented consistently.
func defLineIndent(n *ts.Node, src []byte) string {
	start := n.StartByte()
	ls := start
	for ls > 0 && src[ls-1] != '\n' {
		ls--
	}
	indent := src[ls:start]
	for _, ch := range indent {
		if ch != ' ' && ch != '\t' {
			return ""
		}
	}
	return string(indent)
}

// pyDocstring re-wraps a (first-paragraph, capped) docstring as a triple-quoted
// literal at indent so the skeleton stays valid Python, falling back to a comment
// only in the rare case that both triple-quote styles appear in the text.
func pyDocstring(doc, indent string) string {
	q := `"""`
	if strings.Contains(doc, q) {
		q = `'''`
	}
	if strings.Contains(doc, q) {
		return "# " + strings.ReplaceAll(doc, "\n", "\n"+indent+"# ")
	}
	return q + strings.ReplaceAll(doc, "\n", "\n"+indent) + q
}
