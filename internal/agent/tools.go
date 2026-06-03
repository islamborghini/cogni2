package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/islamborghini/cogni2/internal/meter"
	"github.com/islamborghini/cogni2/internal/retrieve"
	"github.com/islamborghini/cogni2/internal/skeleton"
)

// Tool names the agent loop dispatches on.
const (
	ToolSearchCode   = "search_code"
	ToolReadFile     = "read_file"
	ToolSubmitAnswer = "submit_answer"
)

// Tool is one function the agent can call. Call returns the observation text and
// the meter bucket that text belongs to (retrieved_code for code pulled into
// context, history otherwise).
type Tool interface {
	Spec() ToolSpec
	Call(ctx context.Context, args string) (result, origin string, err error)
}

// --- search_code: Stage 1 retrieval rendered through Stage 2 skeletons ---

type searchCodeTool struct {
	r      retrieve.Retriever
	k      int
	budget int
	tok    skeleton.Tokenizer
}

// NewSearchCodeTool wires the agent's search to the Stage 1 retriever and the
// Stage 2 skeleton assembler, so the end-to-end loop exercises both earlier
// stages as its retrieval layer.
func NewSearchCodeTool(r retrieve.Retriever, k, assemblyBudget int, tok skeleton.Tokenizer) Tool {
	return &searchCodeTool{r: r, k: k, budget: assemblyBudget, tok: tok}
}

func (t *searchCodeTool) Spec() ToolSpec {
	return ToolSpec{
		Name: ToolSearchCode,
		Description: "Search the codebase for code relevant to a natural-language query. " +
			"Returns the top matches: the highest-ranked in full, the rest as signature+docstring " +
			"skeletons each carrying a path:start-end anchor you can re-read with read_file.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string", "description": "what to look for"},
			},
			"required": []string{"query"},
		},
	}
}

func (t *searchCodeTool) Call(ctx context.Context, args string) (string, string, error) {
	var a struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal([]byte(args), &a); err != nil {
		return "", "", fmt.Errorf("search_code: bad args: %w", err)
	}
	if strings.TrimSpace(a.Query) == "" {
		return "", "", fmt.Errorf("search_code: empty query")
	}
	chunks, err := t.r.Retrieve(ctx, a.Query, t.k)
	if err != nil {
		return "", "", err
	}
	asm, err := skeleton.Assemble(chunks, t.budget, t.tok)
	if err != nil {
		return "", "", err
	}
	if asm.Text == "" {
		return "no results", meter.BucketRetrievedCode, nil
	}
	return asm.Text, meter.BucketRetrievedCode, nil
}

// --- read_file: the Stage 2 re-read path ---

type readFileTool struct {
	root     string
	maxLines int
}

// NewReadFileTool reads spans from files under root. maxLines bounds a single
// read so one call cannot dump an entire large file into context.
func NewReadFileTool(root string, maxLines int) Tool {
	if maxLines <= 0 {
		maxLines = 400
	}
	return &readFileTool{root: root, maxLines: maxLines}
}

func (t *readFileTool) Spec() ToolSpec {
	return ToolSpec{
		Name: ToolReadFile,
		Description: "Read a span of lines from a source file (1-based, inclusive). Use the path:start-end " +
			"anchors from search_code results. Only path is required; omit the range to read from the top.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":  map[string]any{"type": "string", "description": "repo-relative file path"},
				"start": map[string]any{"type": "integer", "description": "1-based start line (aliases: start_line, line_start)"},
				"end":   map[string]any{"type": "integer", "description": "1-based end line, inclusive (aliases: end_line, line_end)"},
			},
			"required": []string{"path"},
		},
	}
}

func (t *readFileTool) Call(_ context.Context, args string) (string, string, error) {
	// Accept the common line-range aliases models reach for; only path is required
	// in the schema, so a misnamed range no longer fails provider-side validation.
	var a struct {
		Path      string `json:"path"`
		Start     int    `json:"start"`
		End       int    `json:"end"`
		StartLine int    `json:"start_line"`
		EndLine   int    `json:"end_line"`
		LineStart int    `json:"line_start"`
		LineEnd   int    `json:"line_end"`
	}
	if err := json.Unmarshal([]byte(args), &a); err != nil {
		return "", "", fmt.Errorf("read_file: bad args: %w", err)
	}
	if strings.TrimSpace(a.Path) == "" {
		return "", "", fmt.Errorf("read_file: missing path")
	}
	clean := filepath.Clean(a.Path)
	if filepath.IsAbs(clean) || strings.HasPrefix(clean, "..") {
		return "", "", fmt.Errorf("read_file: path %q escapes the repo", a.Path)
	}
	data, err := os.ReadFile(filepath.Join(t.root, clean))
	if err != nil {
		return "", "", fmt.Errorf("read_file: %w", err)
	}
	lines := strings.Split(string(data), "\n")
	start := firstPositive(a.Start, a.StartLine, a.LineStart)
	if start < 1 {
		start = 1
	}
	if start > len(lines) {
		return "", "", fmt.Errorf("read_file: start %d is past EOF (%d lines)", start, len(lines))
	}
	end := firstPositive(a.End, a.EndLine, a.LineEnd)
	if end <= 0 {
		end = start + t.maxLines - 1 // no range given: read a capped window from start
	}
	if end > len(lines) {
		end = len(lines)
	}
	if end < start {
		end = start
	}
	if end-start+1 > t.maxLines {
		end = start + t.maxLines - 1
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%s:%d-%d\n", clean, start, end)
	for i := start; i <= end; i++ {
		fmt.Fprintf(&b, "%d: %s\n", i, lines[i-1])
	}
	return b.String(), meter.BucketRetrievedCode, nil
}

// --- submit_answer: ends the task; the loop reads the locations ---

// submitAliasCommentary is a second name the loop treats as submit_answer:
// gpt-oss on Groq sometimes emits its internal "commentary" channel as the
// function name on its final call (with valid submit arguments). Advertising it
// (NewSubmitAliasTool) lets the provider accept the call instead of 400ing.
const submitAliasCommentary = "commentary"

// IsSubmitTool reports whether a tool name should finish the task as submit_answer.
func IsSubmitTool(name string) bool {
	return name == ToolSubmitAnswer || name == submitAliasCommentary
}

type submitAnswerTool struct {
	name string
	desc string
}

// NewSubmitAnswerTool returns the tool that finishes a task. The loop special-
// cases it (see IsSubmitTool) to capture the answer; Call itself only acknowledges.
func NewSubmitAnswerTool() Tool {
	return submitAnswerTool{
		name: ToolSubmitAnswer,
		desc: "Submit the final answer: the code location(s) that answer the query, " +
			"as a list of {path, start, end}. Call this once you are confident.",
	}
}

// NewSubmitAliasTool advertises an extra name (e.g. "commentary") that the loop
// also treats as submit_answer — a compatibility shim for models that emit a
// channel name as the final function name.
func NewSubmitAliasTool(name string) Tool {
	return submitAnswerTool{name: name, desc: "Compatibility alias for submit_answer — identical arguments and effect."}
}

func (s submitAnswerTool) Spec() ToolSpec {
	return ToolSpec{
		Name:        s.name,
		Description: s.desc,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"locations": map[string]any{
					"type": "array",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"path":  map[string]any{"type": "string"},
							"start": map[string]any{"type": "integer"},
							"end":   map[string]any{"type": "integer"},
						},
						"required": []string{"path", "start", "end"},
					},
				},
			},
			"required": []string{"locations"},
		},
	}
}

func (submitAnswerTool) Call(_ context.Context, _ string) (string, string, error) {
	return "answer recorded", meter.BucketHistory, nil
}

// parseLocations decodes a submit_answer call's arguments into the answer spans.
func parseLocations(args string) ([]Location, error) {
	var a struct {
		Locations []Location `json:"locations"`
	}
	if err := json.Unmarshal([]byte(args), &a); err != nil {
		return nil, fmt.Errorf("submit_answer: bad args: %w", err)
	}
	if len(a.Locations) == 0 {
		return nil, fmt.Errorf("submit_answer: no locations provided")
	}
	return a.Locations, nil
}

// firstPositive returns the first value greater than zero, or 0 if none are.
func firstPositive(vals ...int) int {
	for _, v := range vals {
		if v > 0 {
			return v
		}
	}
	return 0
}
