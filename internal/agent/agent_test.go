package agent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/islamborghini/cogni2/internal/compress"
	"github.com/islamborghini/cogni2/internal/meter"
	"github.com/islamborghini/cogni2/internal/retrieve"
)

// countTok is a deterministic, offline tokenizer (one token per word). It
// satisfies both meter.Tokenizer and skeleton.Tokenizer, keeping the loop tests
// hermetic.
type countTok struct{}

func (countTok) Count(s string) int { return len(strings.Fields(s)) }

// fakeRetriever returns fixed chunks, standing in for the Stage 1 index.
type fakeRetriever struct{ chunks []retrieve.RetrievedChunk }

func (f fakeRetriever) Retrieve(_ context.Context, _ string, k int) ([]retrieve.RetrievedChunk, error) {
	if k < len(f.chunks) {
		return f.chunks[:k], nil
	}
	return f.chunks, nil
}

func sampleChunks() []retrieve.RetrievedChunk {
	return []retrieve.RetrievedChunk{
		{Path: "pkg/text.py", StartLine: 10, EndLine: 20, Kind: retrieve.KindFunction,
			Content: "def slugify(value):\n    \"\"\"Slugify a string.\"\"\"\n    return re.sub(r\"\\s+\", \"-\", value)"},
	}
}

func searchReadSubmitTools(t *testing.T, root string) []Tool {
	t.Helper()
	return []Tool{
		NewSearchCodeTool(fakeRetriever{sampleChunks()}, 10, 6000, countTok{}),
		NewReadFileTool(root, 400),
		NewSubmitAnswerTool(),
	}
}

func TestRunSubmitsAnswer(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "text.py"), []byte("def slugify(value):\n    return value\n# tail"), 0o644); err != nil {
		t.Fatal(err)
	}
	model := &FakeModel{Responses: []Response{
		{Message: ChatMessage{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: ToolSearchCode, Args: `{"query":"slugify"}`}}}, Usage: Usage{PromptTokens: 50, CompletionTokens: 10}},
		{Message: ChatMessage{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c2", Name: ToolReadFile, Args: `{"path":"text.py","start":1,"end":2}`}}}, Usage: Usage{PromptTokens: 120, CompletionTokens: 12, CachedTokens: 40}},
		{Message: ChatMessage{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c3", Name: ToolSubmitAnswer, Args: `{"locations":[{"path":"pkg/text.py","start":10,"end":20}]}`}}}, Usage: Usage{PromptTokens: 200, CompletionTokens: 8}},
	}}
	deps := Deps{Model: model, Tools: searchReadSubmitTools(t, dir), System: DefaultSystemPrompt, Tok: countTok{}}

	out, tr, ledger, err := Run(context.Background(), RunInput{ID: "t1", Query: "where is slugify defined?"}, deps)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !out.Answered || out.StopReason != "submitted" || out.Turns != 3 {
		t.Fatalf("outcome = %+v, want answered/submitted/3 turns", out)
	}
	if len(out.Locations) != 1 || out.Locations[0].Path != "pkg/text.py" {
		t.Fatalf("locations = %+v, want one pkg/text.py span", out.Locations)
	}
	if len(ledger.Calls) != 3 {
		t.Fatalf("ledger has %d calls, want 3", len(ledger.Calls))
	}
	if ledger.Calls[1].Usage.CachedTokens != 40 {
		t.Errorf("call 1 cached tokens = %d, want 40 (real usage preserved)", ledger.Calls[1].Usage.CachedTokens)
	}
	last := ledger.Calls[2].Buckets
	for _, b := range []string{meter.BucketSystem, meter.BucketHistory, meter.BucketRetrievedCode, meter.BucketOutput} {
		if last[b] <= 0 {
			t.Errorf("final call bucket %q = %d, want > 0", b, last[b])
		}
	}
	// Transcript should end with the submit acknowledgement tool result.
	final := tr.Messages[len(tr.Messages)-1]
	if final.Role != RoleTool || final.ToolCallID != "c3" {
		t.Errorf("last message = %+v, want the c3 tool result", final)
	}
}

func TestRunAcceptsCommentaryAsSubmit(t *testing.T) {
	// gpt-oss sometimes names its final submit call "commentary" — the loop must
	// treat it as submit_answer and capture the locations.
	model := &FakeModel{Responses: []Response{
		{Message: ChatMessage{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: "commentary", Args: `{"locations":[{"path":"pkg/text.py","start":10,"end":20}]}`}}}},
	}}
	deps := Deps{Model: model, Tools: searchReadSubmitTools(t, t.TempDir()), System: DefaultSystemPrompt, Tok: countTok{}}

	out, _, _, err := Run(context.Background(), RunInput{Query: "find it"}, deps)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !out.Answered || out.StopReason != "submitted" || len(out.Locations) != 1 {
		t.Fatalf("commentary call not treated as submit: %+v", out)
	}
}

func TestRunHitsMaxTurns(t *testing.T) {
	model := &FakeModel{Responses: []Response{
		{Message: ChatMessage{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: ToolSearchCode, Args: `{"query":"x"}`}}}},
		{Message: ChatMessage{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c2", Name: ToolSearchCode, Args: `{"query":"y"}`}}}},
		{Message: ChatMessage{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c3", Name: ToolSearchCode, Args: `{"query":"z"}`}}}},
	}}
	deps := Deps{Model: model, Tools: searchReadSubmitTools(t, t.TempDir()), System: DefaultSystemPrompt, Tok: countTok{}, MaxTurns: 3}

	out, _, ledger, err := Run(context.Background(), RunInput{Query: "never answered"}, deps)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if out.Answered || out.StopReason != "max_turns" || out.Turns != 3 || len(ledger.Calls) != 3 {
		t.Fatalf("outcome = %+v, want not-answered/max_turns/3", out)
	}
}

func TestRunNoToolCallEndsLoop(t *testing.T) {
	model := &FakeModel{Responses: []Response{
		{Message: ChatMessage{Role: RoleAssistant, Content: "I am not sure."}},
	}}
	deps := Deps{Model: model, Tools: searchReadSubmitTools(t, t.TempDir()), Tok: countTok{}}

	out, _, _, err := Run(context.Background(), RunInput{Query: "q"}, deps)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if out.Answered || out.StopReason != "no_tool_call" {
		t.Fatalf("outcome = %+v, want not-answered/no_tool_call", out)
	}
}

func TestRunCompressesBetweenSteps(t *testing.T) {
	model := &FakeModel{Responses: []Response{
		{Message: ChatMessage{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: ToolSearchCode, Args: `{"query":"a"}`}}}},
		{Message: ChatMessage{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c2", Name: ToolSearchCode, Args: `{"query":"b"}`}}}},
		{Message: ChatMessage{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c3", Name: ToolSubmitAnswer, Args: `{"locations":[{"path":"pkg/text.py","start":10,"end":20}]}`}}}},
	}}
	comp := &compress.GuidelineCompressor{Summarizer: compress.FakeSummarizer{}, Tok: countTok{}}
	// Tight budget so the accumulated history exceeds it and the compressor engages
	// (gated compression is a no-op under budget — covered by the compress tests).
	deps := Deps{
		Model: model, Tools: searchReadSubmitTools(t, t.TempDir()), System: DefaultSystemPrompt,
		Tok: countTok{}, Compressor: comp, HistoryBudget: 5,
	}

	out, _, _, err := Run(context.Background(), RunInput{Query: "find it"}, deps)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if out.Summarized < 1 {
		t.Fatalf("expected the compressor to summarize an older observation, got %+v", out)
	}
}

func TestMeteringSummarizer(t *testing.T) {
	ms := &MeteringSummarizer{Inner: compress.FakeSummarizer{}, Tok: countTok{}}
	out, err := ms.Summarize(context.Background(), "a/b.py:1-2\nline two\nline three", "preserve anchors")
	if err != nil {
		t.Fatal(err)
	}
	if ms.Calls != 1 || ms.InputTokens <= 0 || ms.OutputTokens <= 0 {
		t.Errorf("metering = %+v, want 1 call and positive in/out tokens", ms)
	}
	if !strings.HasPrefix(out, "a/b.py:1-2") {
		t.Errorf("summary lost its anchor: %q", out)
	}
}
