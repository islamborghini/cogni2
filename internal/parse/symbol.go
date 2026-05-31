// Package parse extracts symbols and references from source files.
//
// v0.1 supports Python only, via the canonical C tree-sitter runtime exposed
// to Go through github.com/tree-sitter/go-tree-sitter (CGO).
package parse

// SymbolKind classifies a parsed symbol. Values are stable strings because
// they are persisted to the on-disk index and surfaced through MCP tools.
type SymbolKind string

const (
	KindFunction SymbolKind = "function"
	KindClass    SymbolKind = "class"
	KindMethod   SymbolKind = "method"
	KindVariable SymbolKind = "variable"
	KindConstant SymbolKind = "constant"
)

// Symbol is a single named definition extracted from a source file.
//
// Line numbers are 1-based and inclusive, matching what users see in editors.
// Qualified is the dotted path within the file (e.g. "Greeter.greet"); the
// indexer prepends the module path at insert time. Signature is the def line
// up to but not including the body, with the trailing colon trimmed; empty
// for classes and assignments. Docstring is the first paragraph of the body's
// docstring if present, truncated to 500 characters.
type Symbol struct {
	Name      string
	Qualified string
	Kind      SymbolKind
	StartLine int
	EndLine   int
	Signature string
	Docstring string
}
