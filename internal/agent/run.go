package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/islamborghini/cogni2/internal/compress"
	"github.com/islamborghini/cogni2/internal/meter"
)

// DefaultMaxTurns caps a single task's loop so a model that never submits cannot
// run forever (and burn the rate-limited free tier).
const DefaultMaxTurns = 12

// maxNoToolCallNudges bounds how many times the loop re-prompts a model that
// answered in prose instead of calling a tool before giving up the task.
const maxNoToolCallNudges = 2

// DefaultSystemPrompt frames the task for the agent. The eval may override it.
const DefaultSystemPrompt = `You are a code navigation assistant working in a large Python codebase.
Find the code that answers the user's question, then report exactly where it is.

Tools:
- search_code(query): semantic search; returns top matches (some in full, some as
  signature+docstring skeletons with a "path:start-end" anchor).
- read_file(path, start, end): read the full source of a span (use the anchors).
- submit_answer(locations): finish by listing the {path,start,end} location(s)
  that answer the question.

Work in small steps: search, read only what you need, then submit. Prefer the
fewest tool calls that let you answer confidently.

You MUST finish by calling submit_answer with the location(s) — do not write the
final answer as plain text. An answer not submitted via submit_answer does not count.`

// The chat client can drive the compressor too.
var _ compress.Summarizer = (*OpenAIChat)(nil)

// Location is one answer span the agent submits; graded against the task gold.
type Location struct {
	Path  string `json:"path"`
	Start int    `json:"start"`
	End   int    `json:"end"`
}

// RunInput is one task to run.
type RunInput struct {
	ID    string
	Query string
}

// Deps are the loop's collaborators. Compressor is nil for the uncompressed
// baseline arm and set for the treatment arm.
type Deps struct {
	Model         Model
	Tools         []Tool
	System        string
	MaxTurns      int
	Tok           meter.Tokenizer
	Compressor    compress.Compressor
	HistoryBudget int
}

// Outcome is the graded-relevant result of a run.
type Outcome struct {
	Answered   bool
	Locations  []Location
	Turns      int
	StopReason string // "submitted" | "max_turns" | "no_tool_call"
	Summarized int    // observations summarized across the run (treatment arm)
	Dropped    int    // observations dropped to fit the budget (treatment arm)
}

// Transcript is the full message log, kept for persistence + offline re-grading.
type Transcript struct {
	Messages []ChatMessage
}

// CallUsage is one model call's accounting: the per-bucket attribution from our
// own tokenizer (the breakdown the report needs) and the API's real billed usage
// (for net cost + the cache-hit-rate guard).
type CallUsage struct {
	Buckets map[string]int
	Usage   Usage
}

// Ledger is the per-call accounting for a whole run.
type Ledger struct {
	Calls []CallUsage
}

// Run drives one task through the loop until the model submits an answer, emits
// no tool call, or hits the turn cap. In the treatment arm it compresses the
// accumulated history after each step.
func Run(ctx context.Context, input RunInput, deps Deps) (Outcome, Transcript, Ledger, error) {
	maxTurns := deps.MaxTurns
	if maxTurns <= 0 {
		maxTurns = DefaultMaxTurns
	}
	byName := make(map[string]Tool, len(deps.Tools))
	specs := make([]ToolSpec, 0, len(deps.Tools))
	for _, t := range deps.Tools {
		byName[t.Spec().Name] = t
		specs = append(specs, t.Spec())
	}
	toolsText := toolsSpecText(specs)

	msgs := []ChatMessage{{Role: RoleUser, Content: input.Query, Origin: meter.BucketHistory}}
	var out Outcome
	var ledger Ledger
	noToolCalls := 0 // consecutive turns where the model answered in prose, not a tool call

	for turn := 0; turn < maxTurns; turn++ {
		resp, err := deps.Model.Generate(ctx, Request{System: deps.System, Messages: msgs, Tools: specs, Temperature: 0})
		if err != nil {
			return out, Transcript{Messages: msgs}, ledger, fmt.Errorf("agent: turn %d: %w", turn, err)
		}
		ledger.Calls = append(ledger.Calls, meterCall(deps.Tok, deps.System, toolsText, msgs, resp))

		assistant := resp.Message
		assistant.Origin = meter.BucketHistory
		msgs = append(msgs, assistant)
		out.Turns = turn + 1

		if len(assistant.ToolCalls) == 0 {
			// Weaker (esp. local) models sometimes reason in prose instead of emitting
			// a tool call. Rather than abandon the task, nudge once or twice and let it
			// try again — this keeps the trajectory going (and recorded) instead of
			// dropping it. Past the cap, accept that it has stopped calling tools.
			noToolCalls++
			if noToolCalls > maxNoToolCallNudges {
				out.StopReason = "no_tool_call"
				return out, Transcript{Messages: msgs}, ledger, nil
			}
			msgs = append(msgs, ChatMessage{
				Role:    RoleUser,
				Content: "Respond with exactly one tool call (search_code, read_file, or submit_answer) — not prose.",
				Origin:  meter.BucketHistory,
			})
			continue
		}
		noToolCalls = 0

		submitted := false
		for _, tc := range assistant.ToolCalls {
			if IsSubmitTool(tc.Name) {
				locs, perr := parseLocations(tc.Args)
				if perr != nil {
					msgs = append(msgs, toolResult(tc.ID, "error: "+perr.Error(), meter.BucketHistory))
					continue
				}
				out.Answered, out.Locations, submitted = true, locs, true
				msgs = append(msgs, toolResult(tc.ID, "answer recorded", meter.BucketHistory))
				continue
			}
			tool := byName[tc.Name]
			if tool == nil {
				msgs = append(msgs, toolResult(tc.ID, "error: unknown tool "+tc.Name, meter.BucketHistory))
				continue
			}
			result, origin, cerr := tool.Call(ctx, tc.Args)
			if cerr != nil {
				result, origin = "error: "+cerr.Error(), meter.BucketHistory
			}
			if origin == "" {
				origin = meter.BucketHistory
			}
			msgs = append(msgs, toolResult(tc.ID, result, origin))
		}
		if submitted {
			out.StopReason = "submitted"
			return out, Transcript{Messages: msgs}, ledger, nil
		}

		if deps.Compressor != nil {
			s, d, cerr := compressMessages(ctx, deps, msgs)
			if cerr != nil {
				return out, Transcript{Messages: msgs}, ledger, cerr
			}
			out.Summarized += s
			out.Dropped += d
		}
	}
	out.StopReason = "max_turns"
	return out, Transcript{Messages: msgs}, ledger, nil
}

func toolResult(id, content, origin string) ChatMessage {
	return ChatMessage{Role: RoleTool, ToolCallID: id, Content: content, Origin: origin}
}

// meterCall attributes one round-trip's tokens to buckets with our own tokenizer:
// system prompt + tool schemas to system, each prior message to its Origin
// (retrieved_code or history), and this turn's generated tokens to output. Summed
// across turns this is the per-bucket view; resp.Usage carries the real billed
// counts for the net-cost guard.
func meterCall(tok meter.Tokenizer, system, toolsText string, msgs []ChatMessage, resp Response) CallUsage {
	cu := CallUsage{Buckets: map[string]int{}, Usage: resp.Usage}
	cu.Buckets[meter.BucketSystem] = tok.Count(system) + tok.Count(toolsText)
	for _, m := range msgs {
		bucket := m.Origin
		if bucket == "" {
			bucket = meter.BucketHistory
		}
		cu.Buckets[bucket] += tok.Count(m.Content) + tok.Count(toolCallsText(m))
	}
	cu.Buckets[meter.BucketOutput] = tok.Count(resp.Message.Content) + tok.Count(toolCallsText(resp.Message))
	return cu
}

func toolCallsText(m ChatMessage) string {
	if len(m.ToolCalls) == 0 {
		return ""
	}
	var b strings.Builder
	for _, tc := range m.ToolCalls {
		b.WriteString(tc.Name)
		b.WriteByte(' ')
		b.WriteString(tc.Args)
		b.WriteByte('\n')
	}
	return b.String()
}

func toolsSpecText(specs []ToolSpec) string {
	b, _ := json.Marshal(specs)
	return string(b)
}

// compressMessages compresses msgs in place. It maps each message to a
// compress.Turn (Content + Role + Origin), runs the compressor, and writes the
// (possibly summarized/dropped) content back by index. Because compress.Compress
// preserves order and length and only edits Content, every tool_call_id linkage
// stays intact.
func compressMessages(ctx context.Context, deps Deps, msgs []ChatMessage) (summarized, dropped int, err error) {
	turns := make([]compress.Turn, len(msgs))
	for i, m := range msgs {
		turns[i] = compress.Turn{Role: m.Role, Content: m.Content, Step: i, Kind: m.Origin, Compressed: m.Compressed}
	}
	res, err := deps.Compressor.Compress(ctx, turns, deps.HistoryBudget)
	if err != nil {
		return 0, 0, err
	}
	// Carry back both the (possibly summarized) content and the Compressed flag so
	// the same observation is never summarized twice across turns.
	for i := range res.History {
		msgs[i].Content = res.History[i].Content
		msgs[i].Compressed = res.History[i].Compressed
	}
	return res.Summarized, res.Dropped, nil
}

// MeteringSummarizer wraps a compress.Summarizer to tally the tokens its calls
// cost — the Stage 3 "compression" overhead the report must charge against the
// treatment arm so the savings are net, not gross.
type MeteringSummarizer struct {
	Inner        compress.Summarizer
	Tok          meter.Tokenizer
	InputTokens  int
	OutputTokens int
	Calls        int
}

// Summarize implements compress.Summarizer.
func (m *MeteringSummarizer) Summarize(ctx context.Context, text, instruction string) (string, error) {
	out, err := m.Inner.Summarize(ctx, text, instruction)
	m.Calls++
	m.InputTokens += m.Tok.Count(text) + m.Tok.Count(instruction)
	if err == nil {
		m.OutputTokens += m.Tok.Count(out)
	}
	return out, err
}
