package parse

import (
	"embed"
	"reflect"
	"testing"
)

//go:embed testdata/*.py
var fixtures embed.FS

func loadFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := fixtures.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatalf("load fixture %s: %v", name, err)
	}
	return data
}

func TestParsePythonTopLevel(t *testing.T) {
	got, err := ParsePythonTopLevel(loadFixture(t, "simple.py"))
	if err != nil {
		t.Fatalf("ParsePythonTopLevel: %v", err)
	}
	want := []Symbol{
		{Name: "hello", Qualified: "hello", Kind: KindFunction, StartLine: 6, EndLine: 8, Signature: "def hello(name)", Docstring: "Greet."},
		{Name: "fetch", Qualified: "fetch", Kind: KindFunction, StartLine: 10, EndLine: 11, Signature: "async def fetch(url)"},
		{Name: "utility", Qualified: "utility", Kind: KindFunction, StartLine: 13, EndLine: 15, Signature: "def utility()"},
		{Name: "Greeter", Qualified: "Greeter", Kind: KindClass, StartLine: 17, EndLine: 19},
		{Name: "Point", Qualified: "Point", Kind: KindClass, StartLine: 21, EndLine: 24},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("symbol mismatch\n got:  %+v\n want: %+v", got, want)
	}
}

func TestParsePython_Simple(t *testing.T) {
	got, err := ParsePython(loadFixture(t, "simple.py"))
	if err != nil {
		t.Fatalf("ParsePython: %v", err)
	}
	want := []Symbol{
		{Name: "CONSTANT", Qualified: "CONSTANT", Kind: KindConstant, StartLine: 4, EndLine: 4},
		{Name: "hello", Qualified: "hello", Kind: KindFunction, StartLine: 6, EndLine: 8, Signature: "def hello(name)", Docstring: "Greet."},
		{Name: "fetch", Qualified: "fetch", Kind: KindFunction, StartLine: 10, EndLine: 11, Signature: "async def fetch(url)"},
		{Name: "utility", Qualified: "utility", Kind: KindFunction, StartLine: 13, EndLine: 15, Signature: "def utility()"},
		{Name: "Greeter", Qualified: "Greeter", Kind: KindClass, StartLine: 17, EndLine: 19},
		{Name: "greet", Qualified: "Greeter.greet", Kind: KindMethod, StartLine: 18, EndLine: 19, Signature: "def greet(self, name)"},
		{Name: "Point", Qualified: "Point", Kind: KindClass, StartLine: 21, EndLine: 24},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("symbol mismatch\n got:  %+v\n want: %+v", got, want)
	}
}

func TestParsePython_Nested(t *testing.T) {
	got, err := ParsePython(loadFixture(t, "nested.py"))
	if err != nil {
		t.Fatalf("ParsePython: %v", err)
	}
	want := []Symbol{
		{Name: "Outer", Qualified: "Outer", Kind: KindClass, StartLine: 3, EndLine: 16, Docstring: "Outer class."},
		{Name: "Inner", Qualified: "Outer.Inner", Kind: KindClass, StartLine: 6, EndLine: 10, Docstring: "Inner class."},
		{Name: "ping", Qualified: "Outer.Inner.ping", Kind: KindMethod, StartLine: 9, EndLine: 10, Signature: "def ping(self)"},
		{Name: "outer_method", Qualified: "Outer.outer_method", Kind: KindMethod, StartLine: 12, EndLine: 16, Signature: "def outer_method(self, x)", Docstring: "Outer method."},
		{Name: "StandAlone", Qualified: "StandAlone", Kind: KindClass, StartLine: 18, EndLine: 19},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("symbol mismatch\n got:  %+v\n want: %+v", got, want)
	}
}

func TestParsePython_Edge(t *testing.T) {
	got, err := ParsePython(loadFixture(t, "edge.py"))
	if err != nil {
		t.Fatalf("ParsePython: %v", err)
	}
	multilineSig := "def multiline_sig(\n    a: int,\n    b: str = \"x\",\n    *args,\n    **kwargs,\n) -> dict[str, int]"
	want := []Symbol{
		{Name: "annotated", Qualified: "annotated", Kind: KindFunction, StartLine: 9, EndLine: 11,
			Signature: `def annotated(a: int, b: str = "x") -> dict[str, int]`, Docstring: "Has annotations."},
		{Name: "gen", Qualified: "gen", Kind: KindFunction, StartLine: 14, EndLine: 16,
			Signature: `async def gen() -> "AsyncIterator[int]"`, Docstring: "Async generator."},
		{Name: "multiline_sig", Qualified: "multiline_sig", Kind: KindFunction, StartLine: 19, EndLine: 26,
			Signature: multilineSig, Docstring: "Wraps onto many lines."},
		{Name: "computed", Qualified: "computed", Kind: KindFunction, StartLine: 29, EndLine: 32,
			Signature: "def computed(self) -> int", Docstring: "Computed value."},
		{Name: "Service", Qualified: "Service", Kind: KindClass, StartLine: 35, EndLine: 44,
			Docstring: "Service."},
		{Name: "make", Qualified: "Service.make", Kind: KindMethod, StartLine: 41, EndLine: 44,
			Signature: `def make(cls, name: str = "x") -> "Service"`, Docstring: "Construct."},
		{Name: "UNICODE_CONST", Qualified: "UNICODE_CONST", Kind: KindConstant, StartLine: 47, EndLine: 47},
		{Name: "mixed_var", Qualified: "mixed_var", Kind: KindVariable, StartLine: 48, EndLine: 48},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("symbol mismatch\n got:  %+v\n want: %+v", got, want)
	}
}

func TestParsePythonTopLevel_Empty(t *testing.T) {
	got, err := ParsePythonTopLevel([]byte("# just a comment\n"))
	if err != nil {
		t.Fatalf("ParsePythonTopLevel: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected no symbols, got %+v", got)
	}
}
