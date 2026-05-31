package parse

// RefKind classifies a reference site. Values mirror the MCP tool schema.
type RefKind string

const (
	RefCall     RefKind = "call"
	RefImport   RefKind = "import"
	RefSubclass RefKind = "subclass"
)

// Reference is a single textual reference to a name at a source location.
//
// Line and Col are 1-based to match what users see in editors. Col is the
// byte column at the start of the identifier. Name is the bare identifier as
// written at the call/import/subclass site (not resolved); v0.1 is textual,
// so a `Foo()` call and a `Foo()` constructor share the same name.
type Reference struct {
	Name string
	Kind RefKind
	Line int
	Col  int
}
