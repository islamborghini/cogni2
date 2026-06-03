package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// defaultLLMBaseURL is Groq's OpenAI-compatible endpoint — the free-tier default
// (Islam is cost-constrained). Override with LLM_BASE_URL to point at OpenRouter,
// Together, a local Ollama server, or OpenAI; the wire shape is identical.
const defaultLLMBaseURL = "https://api.groq.com/openai/v1"

// Retry tuning. Groq's free tier is rate-limited (~30 RPM, low TPM) and returns
// 429 with Retry-After, so we honor it with exponential backoff — the same
// hardening internal/embed needed for Voyage's free tier.
const (
	chatTimeout        = 120 * time.Second
	chatMaxAttempts    = 6
	chatBaseRetryDelay = time.Second
	chatMaxRetryDelay  = 60 * time.Second
	// maxToolUseRetries bounds re-draws on a tool_use_failed 400 (the model emitted
	// an unparseable function call). At temperature ~0 a re-draw often succeeds;
	// past this we give up with the full failed_generation for diagnosis.
	maxToolUseRetries = 3
)

// ErrUnauthorized is returned (wrapped) when the provider rejects the API key
// (HTTP 401/403). It is a global config failure, not a per-task one, so the eval
// aborts on it rather than recording a doomed call for every task.
var ErrUnauthorized = errors.New("agent: unauthorized — check LLM_API_KEY/GROQ_API_KEY")

// OpenAIChat is a provider-agnostic OpenAI-compatible chat-completions client. It
// implements both Model (the agent) and compress.Summarizer (the compressor), so
// the same type — pointed at different models — backs both roles.
type OpenAIChat struct {
	BaseURL string
	APIKey  string
	Model   string
	Client  *http.Client

	// retryBase overrides chatBaseRetryDelay; set small in tests to keep the
	// retry path fast. Zero uses the default.
	retryBase time.Duration
}

// FromEnv builds a chat client for model, reading LLM_BASE_URL (default Groq),
// and the key from LLM_API_KEY or GROQ_API_KEY.
func FromEnv(model string) (*OpenAIChat, error) {
	key := os.Getenv("LLM_API_KEY")
	if key == "" {
		key = os.Getenv("GROQ_API_KEY")
	}
	if key == "" {
		return nil, errors.New("agent: set LLM_API_KEY or GROQ_API_KEY")
	}
	return &OpenAIChat{
		BaseURL: envOr("LLM_BASE_URL", defaultLLMBaseURL),
		APIKey:  key,
		Model:   model,
		Client:  &http.Client{Timeout: chatTimeout},
	}, nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// --- wire shapes (OpenAI chat-completions) ---

type wireToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// wireMessage carries no "name" field on purpose: Groq rejects messages[].name
// with a 400.
type wireMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type wireTool struct {
	Type     string   `json:"type"`
	Function ToolSpec `json:"function"`
}

type wireRequest struct {
	Model       string        `json:"model"`
	Messages    []wireMessage `json:"messages"`
	Tools       []wireTool    `json:"tools,omitempty"`
	ToolChoice  string        `json:"tool_choice,omitempty"`
	Temperature float64       `json:"temperature"`
}

type wireResponse struct {
	Choices []struct {
		Message struct {
			Content   string         `json:"content"`
			ToolCalls []wireToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func toWire(m ChatMessage) wireMessage {
	w := wireMessage{Role: m.Role, Content: m.Content, ToolCallID: m.ToolCallID}
	for _, tc := range m.ToolCalls {
		var wt wireToolCall
		wt.ID, wt.Type = tc.ID, "function"
		wt.Function.Name, wt.Function.Arguments = tc.Name, tc.Args
		w.ToolCalls = append(w.ToolCalls, wt)
	}
	return w
}

// Generate implements Model: one chat-completions round-trip.
func (c *OpenAIChat) Generate(ctx context.Context, req Request) (Response, error) {
	model := req.Model
	if model == "" {
		model = c.Model
	}
	msgs := make([]wireMessage, 0, len(req.Messages)+1)
	if req.System != "" {
		msgs = append(msgs, wireMessage{Role: RoleSystem, Content: req.System})
	}
	for _, m := range req.Messages {
		msgs = append(msgs, toWire(m))
	}
	wr := wireRequest{Model: model, Messages: msgs, Temperature: req.Temperature}
	if len(req.Tools) > 0 {
		wr.ToolChoice = "auto"
		for _, t := range req.Tools {
			wr.Tools = append(wr.Tools, wireTool{Type: "function", Function: t})
		}
	}

	body, err := c.do(ctx, "/chat/completions", wr)
	if err != nil {
		return Response{}, err
	}
	var wresp wireResponse
	if err := json.Unmarshal(body, &wresp); err != nil {
		return Response{}, fmt.Errorf("agent: decode response: %w", err)
	}
	if wresp.Error != nil {
		return Response{}, fmt.Errorf("agent: api error: %s", wresp.Error.Message)
	}
	if len(wresp.Choices) == 0 {
		return Response{}, errors.New("agent: response had no choices")
	}
	ch := wresp.Choices[0]
	msg := ChatMessage{Role: RoleAssistant, Content: ch.Message.Content}
	for _, tc := range ch.Message.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Args: tc.Function.Arguments})
	}
	return Response{
		Message:      msg,
		FinishReason: ch.FinishReason,
		Usage: Usage{
			PromptTokens:     wresp.Usage.PromptTokens,
			CompletionTokens: wresp.Usage.CompletionTokens,
			CachedTokens:     wresp.Usage.PromptTokensDetails.CachedTokens,
		},
	}, nil
}

// Summarize implements compress.Summarizer: the instruction is the system prompt
// and the text is the user turn, with no tools. This lets the same client (a
// cheaper model instance) drive the compressor.
func (c *OpenAIChat) Summarize(ctx context.Context, text, instruction string) (string, error) {
	resp, err := c.Generate(ctx, Request{
		System:   instruction,
		Messages: []ChatMessage{{Role: RoleUser, Content: text}},
	})
	if err != nil {
		return "", err
	}
	return resp.Message.Content, nil
}

// do posts reqBody to BaseURL+path and returns the raw 200 body, retrying 429 and
// 5xx with exponential backoff that honors Retry-After.
func (c *OpenAIChat) do(ctx context.Context, path string, reqBody any) ([]byte, error) {
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	base := c.retryBase
	if base <= 0 {
		base = chatBaseRetryDelay
	}
	client := c.Client
	if client == nil {
		client = &http.Client{Timeout: chatTimeout}
	}

	var lastErr error
	var delay time.Duration
	toolFails := 0
	for attempt := 0; attempt < chatMaxAttempts; attempt++ {
		if delay > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		if c.APIKey != "" {
			req.Header.Set("Authorization", "Bearer "+c.APIKey)
		}
		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			delay = chatBackoff(base, attempt)
			continue
		}
		retryAfter, hasRetryAfter := parseRetryAfter(resp)
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		switch {
		case resp.StatusCode == http.StatusOK:
			return respBody, nil

		case resp.StatusCode == http.StatusBadRequest && bytes.Contains(respBody, []byte("tool_use_failed")):
			// The model emitted a function call the provider could not parse. A
			// re-draw at temperature ~0 often produces a valid call; past the cap we
			// surface the full failed_generation and point at a tool-reliable model.
			toolFails++
			if toolFails > maxToolUseRetries {
				return nil, fmt.Errorf("agent: tool_use_failed after %d attempts (the model produced an unparseable tool call — try a more tool-reliable COGNI_AGENT_MODEL, e.g. openai/gpt-oss-120b): %s", toolFails, snippet(respBody))
			}
			lastErr = fmt.Errorf("agent: tool_use_failed (try %d): %s", toolFails, snippet(respBody))
			delay = chatBackoff(base, toolFails-1)

		case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
			// A bad key won't fix itself across retries or tasks — fail fast and typed.
			return nil, fmt.Errorf("%w: %s: %s", ErrUnauthorized, resp.Status, snippet(respBody))

		case resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500:
			lastErr = fmt.Errorf("agent: %s: %s", resp.Status, snippet(respBody))
			delay = chatBackoff(base, attempt)
			if hasRetryAfter && retryAfter > delay {
				delay = retryAfter
			}
			if delay > chatMaxRetryDelay {
				delay = chatMaxRetryDelay
			}

		default:
			return nil, fmt.Errorf("agent: %s: %s", resp.Status, snippet(respBody))
		}
	}
	return nil, fmt.Errorf("agent: giving up after %d attempts: %w", chatMaxAttempts, lastErr)
}

func chatBackoff(base time.Duration, attempt int) time.Duration {
	d := base << uint(attempt)
	if d > chatMaxRetryDelay {
		d = chatMaxRetryDelay
	}
	return d
}

func parseRetryAfter(resp *http.Response) (time.Duration, bool) {
	v := strings.TrimSpace(resp.Header.Get("Retry-After"))
	if v == "" {
		return 0, false
	}
	if secs, err := strconv.Atoi(v); err == nil && secs >= 0 {
		return time.Duration(secs) * time.Second, true
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d, true
		}
	}
	return 0, false
}

// snippet bounds an error body for the message, but keeps enough to show a
// provider's failed_generation (e.g. Groq's tool_use_failed) for diagnosis.
func snippet(b []byte) string {
	const max = 1500
	if len(b) > max {
		return string(b[:max]) + "…"
	}
	return string(b)
}
