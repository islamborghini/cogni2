package embed

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
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
