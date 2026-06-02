package skeleton

import (
	"strings"
	"testing"

	"github.com/islamborghini/cogni2/internal/parse"
	"github.com/islamborghini/cogni2/internal/retrieve"
)

// parsesClean reports whether src parses with no error nodes — the Stage 2 safety
// property: a skeleton is never broken code.
func parsesClean(t *testing.T, src string) bool {
	t.Helper()
	tree, err := parse.PythonTree([]byte(src))
	if err != nil {
		t.Fatalf("PythonTree: %v", err)
	}
	defer tree.Close()
	return !tree.RootNode().HasError()
}

func TestSkeleton(t *testing.T) {
	tests := []struct {
		name        string
		kind        string
		content     string
		wantContain []string // substrings the skeleton must keep
		wantAbsent  []string // body substrings the skeleton must omit
		verbatim    bool     // expect Content returned unchanged (no parse check)
	}{
		{
			name:        "function with docstring",
			kind:        retrieve.KindFunction,
			content:     "def greet(name):\n    \"\"\"Say hello to name.\"\"\"\n    msg = \"hi \" + name\n    return msg\n",
			wantContain: []string{"def greet(name):", "Say hello to name", "body omitted"},
			wantAbsent:  []string{"\"hi \"", "return msg"},
		},
		{
			name:        "decorated property method",
			kind:        retrieve.KindMethod,
			content:     "@property\ndef value(self):\n    \"\"\"The value.\"\"\"\n    return self._value\n",
			wantContain: []string{"@property", "def value(self):", "The value", "body omitted"},
			wantAbsent:  []string{"self._value"},
		},
		{
			name:        "async with return type, no docstring",
			kind:        retrieve.KindFunction,
			content:     "async def fetch(url) -> str:\n    data = await get(url)\n    return data\n",
			wantContain: []string{"async def fetch(url) -> str:", "body omitted"},
			wantAbsent:  []string{"await get", "return data"},
		},
		{
			name:        "merged two functions",
			kind:        retrieve.KindFunction,
			content:     "def a():\n    return 1\n\n\ndef b():\n    return 2\n",
			wantContain: []string{"def a():", "def b():", "body omitted"},
			wantAbsent:  []string{"return 1", "return 2"},
		},
		{
			name:     "class_header returned unchanged",
			kind:     retrieve.KindClassHeader,
			content:  "class Foo(Base):\n    def m(self) ...\n    def n(self) ...",
			verbatim: true,
		},
		{
			name:     "body fragment passthrough",
			kind:     retrieve.KindFunction,
			content:  "x = 1\ny = compute(x)\n",
			verbatim: true,
		},
		{
			name:     "signature-only fragment passthrough",
			kind:     retrieve.KindMethod,
			content:  "def big(self, a, b, c):\n",
			verbatim: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := retrieve.RetrievedChunk{Path: "m.py", StartLine: 10, EndLine: 20, Kind: tt.kind, Content: tt.content}
			got, err := Skeleton(c)
			if err != nil {
				t.Fatalf("Skeleton: %v", err)
			}
			if tt.verbatim {
				if got != tt.content {
					t.Errorf("expected verbatim passthrough\n got: %q\nwant: %q", got, tt.content)
				}
				return
			}
			for _, want := range tt.wantContain {
				if !strings.Contains(got, want) {
					t.Errorf("skeleton missing %q\n%s", want, got)
				}
			}
			for _, absent := range tt.wantAbsent {
				if strings.Contains(got, absent) {
					t.Errorf("skeleton should have omitted %q\n%s", absent, got)
				}
			}
			if !parsesClean(t, got) {
				t.Errorf("skeleton is not valid Python:\n%s", got)
			}
		})
	}
}

// TestSkeletonPerDefAnchor verifies a merged chunk anchors each definition at its
// true file lines (chunk StartLine + the def's row), not the whole-chunk range.
func TestSkeletonPerDefAnchor(t *testing.T) {
	content := "def a():\n    return 1\n\n\ndef b():\n    return 2\n" // def a at row 0, def b at row 4
	c := retrieve.RetrievedChunk{Path: "m.py", StartLine: 100, EndLine: 106, Kind: retrieve.KindFunction, Content: content}
	got, err := Skeleton(c)
	if err != nil {
		t.Fatalf("Skeleton: %v", err)
	}
	for _, want := range []string{"expand via m.py:100-", "expand via m.py:104-"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing precise per-def anchor %q\n%s", want, got)
		}
	}
}
