package agent

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestOpenAIChatGenerateParsesToolCallsAndUsage(t *testing.T) {
	const body = `{"choices":[{"message":{"content":"","tool_calls":[` +
		`{"id":"call_1","type":"function","function":{"name":"search_code","arguments":"{\"query\":\"slugify\"}"}}]},` +
		`"finish_reason":"tool_calls"}],` +
		`"usage":{"prompt_tokens":100,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":80}}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer secret" {
			t.Errorf("auth header = %q, want bearer", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	defer srv.Close()

	c := &OpenAIChat{BaseURL: srv.URL, APIKey: "secret", Model: "test-model", Client: srv.Client()}
	resp, err := c.Generate(context.Background(), Request{
		Messages: []ChatMessage{{Role: RoleUser, Content: "where is slugify"}},
		Tools:    []ToolSpec{{Name: "search_code"}},
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.Name != "search_code" || tc.ID != "call_1" || tc.Args != `{"query":"slugify"}` {
		t.Errorf("tool call = %+v, want parsed search_code", tc)
	}
	if resp.FinishReason != "tool_calls" {
		t.Errorf("finish reason = %q", resp.FinishReason)
	}
	if resp.Usage != (Usage{PromptTokens: 100, CompletionTokens: 20, CachedTokens: 80}) {
		t.Errorf("usage = %+v, want 100/20/80 incl cached", resp.Usage)
	}
}

func TestOpenAIChatRetriesOn429(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"message":"rate limited"}}`)
			return
		}
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer srv.Close()

	c := &OpenAIChat{BaseURL: srv.URL, APIKey: "x", Model: "m", Client: srv.Client(), retryBase: time.Millisecond}
	resp, err := c.Generate(context.Background(), Request{Messages: []ChatMessage{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Message.Content != "ok" {
		t.Errorf("content = %q, want ok after retry", resp.Message.Content)
	}
	if n := atomic.LoadInt32(&calls); n != 2 {
		t.Errorf("server calls = %d, want 2 (one 429 then success)", n)
	}
}

func TestOpenAIChatErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"messages.0.name unsupported"}}`)
	}))
	defer srv.Close()

	c := &OpenAIChat{BaseURL: srv.URL, APIKey: "x", Model: "m", Client: srv.Client()}
	if _, err := c.Generate(context.Background(), Request{Messages: []ChatMessage{{Role: RoleUser, Content: "hi"}}}); err == nil {
		t.Error("a 400 should surface as an error, not be retried")
	}
}
