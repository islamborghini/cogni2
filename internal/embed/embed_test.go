package embed

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestFake_Deterministic(t *testing.T) {
	f := Fake{Dim: 32}
	a, err := f.Embed(context.Background(), []string{"hello", "world", "hello"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(a) != 3 {
		t.Fatalf("want 3 vectors, got %d", len(a))
	}
	for _, v := range a {
		if len(v) != 32 {
			t.Errorf("want dim 32, got %d", len(v))
		}
		if n := norm(v); math.Abs(float64(n)-1) > 1e-5 {
			t.Errorf("want unit vector, got norm %v", n)
		}
	}
	if !equal(a[0], a[2]) {
		t.Error("same text should map to the same vector")
	}
	if equal(a[0], a[1]) {
		t.Error("different texts should map to different vectors")
	}
}

func TestVoyage_BatchesAndOrders(t *testing.T) {
	var requests int32
	var gotInputType string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requests, 1)
		var req voyageRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		gotInputType = req.InputType

		// Respond in REVERSE index order to prove the client sorts by index.
		// Each embedding encodes the input's length so we can verify mapping.
		type item struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}
		items := make([]item, len(req.Input))
		for i, s := range req.Input {
			items[len(req.Input)-1-i] = item{Embedding: []float32{float32(len(s))}, Index: i}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": items})
	}))
	defer srv.Close()

	v := &Voyage{
		APIKey:    "test",
		Model:     "voyage-code-3",
		Endpoint:  srv.URL,
		InputType: "document",
		BatchSize: 2,
	}
	texts := []string{"a", "bb", "ccc", "dddd", "eeeee"}
	got, err := v.Embed(context.Background(), texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(got) != len(texts) {
		t.Fatalf("want %d vectors, got %d", len(texts), len(got))
	}
	for i, s := range texts {
		if got[i][0] != float32(len(s)) {
			t.Errorf("vector %d mismatched (ordering lost): want %d, got %v", i, len(s), got[i][0])
		}
	}
	if n := atomic.LoadInt32(&requests); n != 3 { // ceil(5/2)
		t.Errorf("want 3 batched requests, got %d", n)
	}
	if gotInputType != "document" {
		t.Errorf("input_type not sent: %q", gotInputType)
	}
}

func TestParseRetryAfter(t *testing.T) {
	mk := func(v string) *http.Response {
		return &http.Response{Header: http.Header{"Retry-After": []string{v}}}
	}
	if d, ok := parseRetryAfter(mk("5")); !ok || d != 5*time.Second {
		t.Errorf("seconds form: got %v %v", d, ok)
	}
	future := time.Now().Add(3 * time.Second).UTC().Format(http.TimeFormat)
	if d, ok := parseRetryAfter(mk(future)); !ok || d <= 0 {
		t.Errorf("http-date form: got %v %v", d, ok)
	}
	if _, ok := parseRetryAfter(&http.Response{Header: http.Header{}}); ok {
		t.Error("missing header should report false")
	}
}

func TestVoyage_RetriesThenSucceeds(t *testing.T) {
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&n, 1) == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"detail":"slow down"}`))
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"embedding": []float32{1, 2}, "index": 0}},
		})
	}))
	defer srv.Close()

	v := &Voyage{APIKey: "k", Model: "voyage-code-3", Endpoint: srv.URL, InputType: "document"}
	got, err := v.Embed(context.Background(), []string{"x"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(got) != 1 || len(got[0]) != 2 {
		t.Fatalf("unexpected result: %v", got)
	}
	if atomic.LoadInt32(&n) != 2 {
		t.Errorf("want 2 requests (429 then 200), got %d", n)
	}
}

func TestFromEnv(t *testing.T) {
	t.Setenv("EMBED_PROVIDER", "ollama")
	t.Setenv("EMBED_MODEL", "nomic-embed-text")
	e, err := FromEnv("document")
	if err != nil {
		t.Fatalf("FromEnv ollama: %v", err)
	}
	if _, ok := e.(*OpenAICompatible); !ok {
		t.Errorf("want *OpenAICompatible, got %T", e)
	}

	t.Setenv("EMBED_PROVIDER", "voyage")
	t.Setenv("VOYAGE_API_KEY", "k")
	if e, err := FromEnv("query"); err != nil {
		t.Fatalf("FromEnv voyage: %v", err)
	} else if _, ok := e.(*Voyage); !ok {
		t.Errorf("want *Voyage, got %T", e)
	}

	t.Setenv("EMBED_PROVIDER", "nope")
	if _, err := FromEnv("document"); err == nil {
		t.Error("unknown provider should error")
	}
}

func TestBatched_TokenAware(t *testing.T) {
	// Each text is ~5 estimated tokens (20 bytes / 4). With maxTokens=12, at most
	// two fit per batch.
	texts := []string{
		strings.Repeat("a", 20), strings.Repeat("b", 20),
		strings.Repeat("c", 20), strings.Repeat("d", 20), strings.Repeat("e", 20),
	}
	var sizes []int
	_, err := batched(context.Background(), texts, 100, 12, func(_ context.Context, batch []string) ([][]float32, error) {
		sizes = append(sizes, len(batch))
		out := make([][]float32, len(batch))
		for i := range out {
			out[i] = []float32{0}
		}
		return out, nil
	})
	if err != nil {
		t.Fatalf("batched: %v", err)
	}
	for _, s := range sizes {
		if s > 2 {
			t.Errorf("batch of %d exceeds the token budget; sizes=%v", s, sizes)
		}
	}
}

func norm(v []float32) float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return float32(math.Sqrt(s))
}

func equal(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
