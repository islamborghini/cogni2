package parse

import (
	"errors"

	ts "github.com/tree-sitter/go-tree-sitter"
)

// ParsePythonRefs returns every reference site in src: function calls, names
// brought in by `import`/`from x import y`, and superclasses on class
// definitions. Attribute access (`obj.foo`) is not emitted in v0.1 — it would
// flood results without semantic disambiguation.
func ParsePythonRefs(src []byte) ([]Reference, error) {
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

	var out []Reference
	walkRefs(tree.RootNode(), src, &out)
	return out, nil
}

func walkRefs(n *ts.Node, src []byte, out *[]Reference) {
	switch n.Kind() {
	case "call":
		emitCallRef(n, src, out)
	case "class_definition":
		emitSubclassRefs(n, src, out)
	case "import_statement", "import_from_statement":
		emitImportRefs(n, src, out)
	}
	for i := uint(0); i < n.NamedChildCount(); i++ {
		walkRefs(n.NamedChild(i), src, out)
	}
}

// emitCallRef pulls the identifier off a call's `function` field. For an
// attribute call like `a.b.foo()` we emit the rightmost name (`foo`), since
// that's what users mean by "find references to foo".
func emitCallRef(call *ts.Node, src []byte, out *[]Reference) {
	fn := call.ChildByFieldName("function")
	if fn == nil {
		return
	}
	id := terminalIdent(fn)
	if id == nil {
		return
	}
	*out = append(*out, Reference{
		Name: id.Utf8Text(src),
		Kind: RefCall,
		Line: int(id.StartPosition().Row) + 1,
		Col:  int(id.StartPosition().Column) + 1,
	})
}

// emitSubclassRefs emits one ref per superclass identifier. `class Foo(Bar, mod.Baz)`
// emits Bar (subclass) and Baz (subclass).
func emitSubclassRefs(cls *ts.Node, src []byte, out *[]Reference) {
	args := cls.ChildByFieldName("superclasses")
	if args == nil {
		return
	}
	for i := uint(0); i < args.NamedChildCount(); i++ {
		c := args.NamedChild(i)
		id := terminalIdent(c)
		if id == nil {
			continue
		}
		*out = append(*out, Reference{
			Name: id.Utf8Text(src),
			Kind: RefSubclass,
			Line: int(id.StartPosition().Row) + 1,
			Col:  int(id.StartPosition().Column) + 1,
		})
	}
}

// emitImportRefs handles both `import a, b.c` and `from x import y, z as w`.
// For `from x import y` we emit y. For `import a.b` we emit the leaf (b).
// `as` aliases emit the original name (the thing being referenced), not the
// alias.
func emitImportRefs(imp *ts.Node, src []byte, out *[]Reference) {
	for i := uint(0); i < imp.NamedChildCount(); i++ {
		c := imp.NamedChild(i)
		var target *ts.Node
		switch c.Kind() {
		case "dotted_name", "identifier":
			// Skip the module name on `from x import …` — handled per-name below.
			if imp.Kind() == "import_from_statement" {
				if mod := imp.ChildByFieldName("module_name"); mod != nil && c.Equals(*mod) {
					continue
				}
			}
			target = c
		case "aliased_import":
			target = c.ChildByFieldName("name")
		default:
			continue
		}
		id := terminalIdent(target)
		if id == nil {
			continue
		}
		*out = append(*out, Reference{
			Name: id.Utf8Text(src),
			Kind: RefImport,
			Line: int(id.StartPosition().Row) + 1,
			Col:  int(id.StartPosition().Column) + 1,
		})
	}
}

// terminalIdent returns the rightmost identifier of an `identifier` or
// `attribute`/`dotted_name` chain, or nil for shapes we don't handle.
func terminalIdent(n *ts.Node) *ts.Node {
	if n == nil {
		return nil
	}
	switch n.Kind() {
	case "identifier":
		return n
	case "attribute":
		return n.ChildByFieldName("attribute")
	case "dotted_name":
		// rightmost named child is the leaf
		if n.NamedChildCount() == 0 {
			return nil
		}
		return n.NamedChild(n.NamedChildCount() - 1)
	}
	return nil
}
