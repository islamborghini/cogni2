// Package embed turns text into vectors behind a single Embedder interface, so
// the retrieval layer never hardcodes a provider.
//
// Stage 1 ships three implementations: Voyage (the code-specialized default),
// OpenAICompatible (any /v1/embeddings endpoint, e.g. a local Ollama server),
// and Fake (deterministic, offline) for hermetic tests.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Embedder maps a batch of texts to their vectors, preserving order. A nil error
// guarantees len(result) == len(texts).
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// defaultBatchSize bounds how many texts go in one HTTP request.
const defaultBatchSize = 128

// batched applies fn over texts in batches and concatenates the results. A batch
// is capped at maxItems texts and, when maxTokens > 0, at roughly maxTokens
// estimated tokens — the latter keeps a single request under a provider's
// per-minute token ceiling (e.g. Voyage's free tier). A text larger than the
// token cap is still sent on its own.
func batched(ctx context.Context, texts []string, maxItems, maxTokens int, fn func(context.Context, []string) ([][]float32, error)) ([][]float32, error) {
	if maxItems <= 0 {
		maxItems = defaultBatchSize
	}
	out := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); {
		j, tokens := i, 0
		for j < len(texts) && j-i < maxItems {
			t := estTokens(texts[j])
			if maxTokens > 0 && j > i && tokens+t > maxTokens {
				break
			}
			tokens += t
			j++
		}
		if j == i { // single oversized text: send it alone
			j = i + 1
		}
		vecs, err := fn(ctx, texts[i:j])
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
		i = j
	}
	return out, nil
}

// estTokens is a cheap byte-based token estimate (~4 bytes/token) used only to
// size batches; exact counting is the meter's job, not the embedder's.
func estTokens(s string) int { return len(s)/4 + 1 }

// embedResponse is the shared OpenAI/Voyage embeddings response shape.
type embedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

// Retry tuning. A 429 with a Retry-After header is honored up to maxRetryDelay,
// which matters on rate-limited tiers (e.g. Voyage free at 3 RPM).
const (
	maxAttempts    = 6
	maxRetryDelay  = 60 * time.Second
	baseRetryDelay = time.Second
)

// postEmbeddings sends reqBody to endpoint as JSON with a bearer token, retrying
// transient failures (429 and 5xx) with exponential backoff that honors a
// Retry-After header, and returns the embeddings ordered by their response index.
func postEmbeddings(ctx context.Context, client *http.Client, endpoint, apiKey string, reqBody any) ([][]float32, error) {
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	var lastErr error
	var delay time.Duration
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if delay > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		if apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+apiKey)
		}

		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			delay = backoff(attempt)
			continue
		}
		retryAfter, hasRetryAfter := parseRetryAfter(resp)
		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("embed: %s: %s", resp.Status, snippet(body))
			delay = backoff(attempt)
			if hasRetryAfter && retryAfter > delay {
				delay = retryAfter
			}
			if delay > maxRetryDelay {
				delay = maxRetryDelay
			}
			continue
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("embed: %s: %s", resp.Status, snippet(body))
		}

		var er embedResponse
		if err := json.Unmarshal(body, &er); err != nil {
			return nil, fmt.Errorf("embed: decode response: %w", err)
		}
		sort.Slice(er.Data, func(i, j int) bool { return er.Data[i].Index < er.Data[j].Index })
		out := make([][]float32, len(er.Data))
		for i := range er.Data {
			out[i] = er.Data[i].Embedding
		}
		return out, nil
	}
	return nil, fmt.Errorf("embed: giving up after %d attempts: %w", maxAttempts, lastErr)
}

// backoff returns the exponential delay for a given attempt, capped.
func backoff(attempt int) time.Duration {
	d := baseRetryDelay << uint(attempt)
	if d > maxRetryDelay {
		d = maxRetryDelay
	}
	return d
}

// parseRetryAfter reads a Retry-After header in either delta-seconds or HTTP-date
// form.
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

func snippet(b []byte) string {
	const max = 200
	if len(b) > max {
		return string(b[:max]) + "…"
	}
	return string(b)
}

// Fake is a deterministic, offline Embedder: the same text always maps to the
// same unit vector, and different texts map to different ones. It backs every
// hermetic test in the project so they need no network or API key.
type Fake struct {
	Dim int // defaults to 64
}

// Embed returns one seeded unit vector per text.
func (f Fake) Embed(_ context.Context, texts []string) ([][]float32, error) {
	dim := f.Dim
	if dim <= 0 {
		dim = 64
	}
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = seededUnitVec(t, dim)
	}
	return out, nil
}

func seededUnitVec(text string, dim int) []float32 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(text))
	rng := rand.New(rand.NewSource(int64(h.Sum64())))
	v := make([]float32, dim)
	var norm float64
	for i := range v {
		x := rng.Float64()*2 - 1
		v[i] = float32(x)
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		norm = 1
	}
	for i := range v {
		v[i] = float32(float64(v[i]) / norm)
	}
	return v
}
