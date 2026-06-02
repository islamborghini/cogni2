package agent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/islamborghini/cogni2/internal/meter"
)

func TestSearchCodeToolRendersAnchoredResults(t *testing.T) {
	tool := NewSearchCodeTool(fakeRetriever{sampleChunks()}, 10, 6000, countTok{})
	res, origin, err := tool.Call(context.Background(), `{"query":"slugify"}`)
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if origin != meter.BucketRetrievedCode {
		t.Errorf("origin = %q, want retrieved_code", origin)
	}
	if !strings.Contains(res, "pkg/text.py:10-20") {
		t.Errorf("result missing the re-read anchor:\n%s", res)
	}
}

func TestSearchCodeToolBadArgs(t *testing.T) {
	tool := NewSearchCodeTool(fakeRetriever{sampleChunks()}, 10, 6000, countTok{})
	if _, _, err := tool.Call(context.Background(), `{`); err == nil {
		t.Error("malformed JSON should error")
	}
	if _, _, err := tool.Call(context.Background(), `{"query":"  "}`); err == nil {
		t.Error("empty query should error")
	}
}

func TestReadFileTool(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "x.py"), []byte("l1\nl2\nl3\nl4\nl5"), 0o644); err != nil {
		t.Fatal(err)
	}
	tool := NewReadFileTool(dir, 400)

	res, origin, err := tool.Call(context.Background(), `{"path":"x.py","start":2,"end":4}`)
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if origin != meter.BucketRetrievedCode {
		t.Errorf("origin = %q, want retrieved_code", origin)
	}
	for _, want := range []string{"x.py:2-4", "2: l2", "4: l4"} {
		if !strings.Contains(res, want) {
			t.Errorf("read result missing %q:\n%s", want, res)
		}
	}
	if strings.Contains(res, "1: l1") || strings.Contains(res, "5: l5") {
		t.Errorf("read result leaked out-of-range lines:\n%s", res)
	}

	if _, _, err := tool.Call(context.Background(), `{"path":"../escape.py","start":1,"end":1}`); err == nil {
		t.Error("path escaping the repo should error")
	}
	if _, _, err := tool.Call(context.Background(), `{"path":"missing.py","start":1,"end":1}`); err == nil {
		t.Error("missing file should error")
	}
	if _, _, err := tool.Call(context.Background(), `{"path":"x.py","start":100,"end":101}`); err == nil {
		t.Error("start past EOF should error")
	}
}

func TestParseLocations(t *testing.T) {
	locs, err := parseLocations(`{"locations":[{"path":"a.py","start":1,"end":3},{"path":"b.py","start":5,"end":9}]}`)
	if err != nil || len(locs) != 2 || locs[1].Path != "b.py" || locs[1].End != 9 {
		t.Fatalf("parseLocations = (%+v, %v), want two parsed spans", locs, err)
	}
	if _, err := parseLocations(`{"locations":[]}`); err == nil {
		t.Error("empty locations should error")
	}
	if _, err := parseLocations(`not json`); err == nil {
		t.Error("malformed JSON should error")
	}
}
