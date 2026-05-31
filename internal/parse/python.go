package parse

import (
	"errors"
	"strings"
	"unicode/utf8"

	ts "github.com/tree-sitter/go-tree-sitter"
	tspython "github.com/tree-sitter/tree-sitter-python/bindings/go"
)

// docstringMaxLen caps the stored docstring length. Matches the SQLite schema
// reasoning in the v0.1 plan: docstrings are summary-grade, not full text.
const docstringMaxLen = 500

// pythonLang is loaded once. tree-sitter Languages are immutable and safe for
// concurrent reuse, unlike Parser instances.
var pythonLang = ts.NewLanguage(tspython.Language())

// ParsePython returns every named definition (function, class, method) in src.
// Qualified names are dotted paths within the file: a top-level function gets
// Qualified == Name; a method on Greeter gets Qualified == "Greeter.greet".
//
// ParsePythonTopLevel is preserved for callers that only want module-level
// items; it is now a thin filter over ParsePython.
func ParsePython(src []byte) ([]Symbol, error) {
	parser := ts.NewParser()
	defer parser.Close()
	if err := parser.SetLanguage(pythonLang); err != nil {
		return nil, err
	}
	tree := parser.Parse(src, nil)
	if tree == nil {
		return nil, errors.New("parse: tree-sitter returned nil tree")
	}
	defer tree.Close()

	var out []Symbol
	walkScope(tree.RootNode(), src, "", false, &out)
	return out, nil
}

// ParsePythonTopLevel returns only the module-level functions and classes.
func ParsePythonTopLevel(src []byte) ([]Symbol, error) {
	all, err := ParsePython(src)
	if err != nil {
		return nil, err
	}
	out := all[:0]
	for _, s := range all {
		if s.Kind == KindFunction || s.Kind == KindClass {
			if s.Qualified == s.Name { // no dotted prefix → top-level
				out = append(out, s)
			}
		}
	}
	return out, nil
}

// walkScope visits a node's named children and emits symbols. parent is the
// qualified-name prefix ("" at module level, "Greeter" inside a class body).
// inClass flips KindFunction → KindMethod for definitions found one level
// inside a class.
func walkScope(scope *ts.Node, src []byte, parent string, inClass bool, out *[]Symbol) {
	for i := uint(0); i < scope.NamedChildCount(); i++ {
		c := scope.NamedChild(i)
		switch c.Kind() {
		case "function_definition", "decorated_definition", "class_definition":
			sym, body, ok := extractDef(c, src, parent)
			if !ok {
				continue
			}
			if inClass && sym.Kind == KindFunction {
				sym.Kind = KindMethod
			}
			*out = append(*out, sym)
			if sym.Kind == KindClass && body != nil {
				walkScope(body, src, sym.Qualified, true, out)
			}
		case "expression_statement":
			if inClass || c.NamedChildCount() == 0 {
				continue
			}
			inner := c.NamedChild(0)
			if inner.Kind() != "assignment" {
				continue
			}
			if sym, ok := extractAssign(inner, src, parent); ok {
				*out = append(*out, sym)
			}
		}
	}
}

// extractAssign turns a simple `name = value` or `NAME: T = value` assignment
// into a Symbol. Tuple unpacking, attribute targets (`obj.x = ...`), and
// subscript targets are skipped at this layer.
func extractAssign(n *ts.Node, src []byte, parent string) (Symbol, bool) {
	left := n.ChildByFieldName("left")
	if left == nil || left.Kind() != "identifier" {
		return Symbol{}, false
	}
	name := left.Utf8Text(src)
	kind := KindVariable
	if isAllCaps(name) {
		kind = KindConstant
	}
	qualified := name
	if parent != "" {
		qualified = parent + "." + name
	}
	return Symbol{
		Name:      name,
		Qualified: qualified,
		Kind:      kind,
		StartLine: int(n.StartPosition().Row) + 1,
		EndLine:   int(n.EndPosition().Row) + 1,
	}, true
}

// isAllCaps reports whether s contains at least one ASCII letter and no
// lowercase letters — the conventional Python marker for a module-level
// constant. Underscores and digits are neutral.
func isAllCaps(s string) bool {
	hasLetter := false
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z':
			return false
		case r >= 'A' && r <= 'Z':
			hasLetter = true
		}
	}
	return hasLetter
}

// extractDef pulls a Symbol out of a function_definition, class_definition, or
// decorated_definition node and also returns the body block (for classes, so
// the caller can recurse). ok=false for unsupported node kinds or malformed
// definitions missing a name.
func extractDef(n *ts.Node, src []byte, parent string) (Symbol, *ts.Node, bool) {
	startLine := int(n.StartPosition().Row) + 1
	endLine := int(n.EndPosition().Row) + 1

	def := n
	if n.Kind() == "decorated_definition" {
		def = n.ChildByFieldName("definition")
		if def == nil {
			return Symbol{}, nil, false
		}
	}

	var kind SymbolKind
	switch def.Kind() {
	case "function_definition":
		kind = KindFunction
	case "class_definition":
		kind = KindClass
	default:
		return Symbol{}, nil, false
	}

	nameNode := def.ChildByFieldName("name")
	if nameNode == nil {
		return Symbol{}, nil, false
	}
	name := nameNode.Utf8Text(src)
	qualified := name
	if parent != "" {
		qualified = parent + "." + name
	}

	sym := Symbol{
		Name:      name,
		Qualified: qualified,
		Kind:      kind,
		StartLine: startLine,
		EndLine:   endLine,
	}

	body := def.ChildByFieldName("body")
	if kind == KindFunction {
		sym.Signature = extractSignature(def, body, src)
	}
	sym.Docstring = extractDocstring(body, src)
	if kind == KindClass {
		return sym, body, true
	}
	return sym, nil, true
}

// extractDocstring returns the first-paragraph docstring for the given body
// block, or "" if the first statement is not a bare string. The result is
// trimmed, cut at the first blank line, and capped at docstringMaxLen runes.
func extractDocstring(body *ts.Node, src []byte) string {
	if body == nil || body.NamedChildCount() == 0 {
		return ""
	}
	first := body.NamedChild(0)
	if first.Kind() != "expression_statement" || first.NamedChildCount() == 0 {
		return ""
	}
	str := first.NamedChild(0)
	if str.Kind() != "string" {
		return ""
	}

	var b strings.Builder
	for i := uint(0); i < str.NamedChildCount(); i++ {
		c := str.NamedChild(i)
		if c.Kind() == "string_content" {
			b.WriteString(c.Utf8Text(src))
		}
	}
	raw := b.String()
	if idx := strings.Index(raw, "\n\n"); idx >= 0 {
		raw = raw[:idx]
	}
	raw = strings.TrimSpace(raw)
	if utf8.RuneCountInString(raw) > docstringMaxLen {
		runes := []rune(raw)
		raw = string(runes[:docstringMaxLen])
	}
	return raw
}

// extractSignature returns the def line up to (but excluding) the body, with
// trailing whitespace and the colon trimmed. Returns "def hello(name)" for
// `def hello(name):\n    ...` and "async def fetch(url) -> str" for the typed
// async form. Returns empty string if the body offset can't be determined.
func extractSignature(def, body *ts.Node, src []byte) string {
	if body == nil {
		return ""
	}
	start := def.StartByte()
	end := body.StartByte()
	if end <= start || end > uint(len(src)) {
		return ""
	}
	sig := strings.TrimRight(string(src[start:end]), " \t\n:")
	return sig
}
