// Package agent runs the multi-turn loop that Stage 3 measures: a model calls
// tools (code search, file read, answer submission), observations accumulate as
// history, and — in the treatment arm — a compress.Compressor condenses older
// observations between steps. This is the first executable agent in the project;
// Stages 1–2 were measured offline without a loop.
//
// The model is reached through a provider-agnostic OpenAI-compatible chat client
// (Groq by default), mirroring internal/embed: no SDK, hand-rolled net/http with
// Retry-After backoff. A deterministic FakeModel backs the hermetic loop tests.
package agent

import "context"

// Chat roles on the wire (OpenAI chat-completions).
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// ToolCall is one function call the model requested.
type ToolCall struct {
	ID   string
	Name string
	Args string // raw JSON arguments, exactly as the model emitted them
}

// ChatMessage is one message in the conversation. Origin records which meter
// bucket this message's tokens belong to when it is re-sent as input on later
// turns (retrieved_code vs history); it is internal bookkeeping, never sent to
// the API.
type ChatMessage struct {
	Role       string
	Content    string
	ToolCalls  []ToolCall
	ToolCallID string
	Origin     string
	// Compressed marks an observation the compressor has already summarized, so it
	// is not summarized again on later turns (incremental compression).
	Compressed bool
}

// ToolSpec is a function tool advertised to the model; it is serialized as each
// request's tools[].function.
type ToolSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// Usage is the billed token accounting one model call reports. CachedTokens is
// the prefix-cache hit (Groq: usage.prompt_tokens_details.cached_tokens), the
// field the Stage 3 net-cost guard is built on.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	CachedTokens     int
}

// Request is one model round-trip.
type Request struct {
	Model       string // overrides the client default when set
	System      string
	Messages    []ChatMessage
	Tools       []ToolSpec
	Temperature float64
}

// Response is the assistant turn plus its billed usage.
type Response struct {
	Message      ChatMessage
	Usage        Usage
	FinishReason string
}

// Model is one chat-completions round-trip, behind an interface so the real
// OpenAI-compatible client and the deterministic FakeModel are swappable.
type Model interface {
	Generate(ctx context.Context, req Request) (Response, error)
}
