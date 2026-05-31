package embed

import (
	"context"
	"errors"
	"net/http"
	"os"
	"time"
)

// Default endpoints and model for the code-specialized default provider.
const (
	defaultVoyageEndpoint = "https://api.voyageai.com/v1/embeddings"
	defaultVoyageModel    = "voyage-code-3"
)

// httpTimeout bounds a single embeddings request.
const httpTimeout = 60 * time.Second

// Voyage embeds via Voyage AI's API. voyage-code-3 is code-specialized and has a
// large free token allotment, which makes it the Stage 1 default. InputType must
// be "document" when embedding chunks and "query" when embedding a search query;
// Voyage uses it to encode the two sides of the retrieval pair differently.
type Voyage struct {
	APIKey    string
	Model     string
	Endpoint  string
	InputType string // "document" | "query"
	Dim       int    // output_dimension; 0 leaves the model default
	BatchSize int
	Client    *http.Client
}

type voyageRequest struct {
	Input           []string `json:"input"`
	Model           string   `json:"model"`
	InputType       string   `json:"input_type,omitempty"`
	OutputDimension int      `json:"output_dimension,omitempty"`
}

// NewVoyage builds a Voyage embedder for the given input type ("document" or
// "query"), reading VOYAGE_API_KEY, EMBED_MODEL, and EMBED_ENDPOINT from the
// environment and falling back to the code-specialized defaults.
func NewVoyage(inputType string) (*Voyage, error) {
	key := os.Getenv("VOYAGE_API_KEY")
	if key == "" {
		return nil, errors.New("embed: VOYAGE_API_KEY is not set")
	}
	return &Voyage{
		APIKey:    key,
		Model:     envOr("EMBED_MODEL", defaultVoyageModel),
		Endpoint:  envOr("EMBED_ENDPOINT", defaultVoyageEndpoint),
		InputType: inputType,
		Client:    &http.Client{Timeout: httpTimeout},
	}, nil
}

// Embed implements Embedder.
func (v *Voyage) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	client := v.Client
	if client == nil {
		client = &http.Client{Timeout: httpTimeout}
	}
	return batched(ctx, texts, v.BatchSize, func(ctx context.Context, batch []string) ([][]float32, error) {
		return postEmbeddings(ctx, client, v.Endpoint, v.APIKey, voyageRequest{
			Input:           batch,
			Model:           v.Model,
			InputType:       v.InputType,
			OutputDimension: v.Dim,
		})
	})
}

// OpenAICompatible embeds via any endpoint that speaks the OpenAI
// /v1/embeddings shape — OpenAI itself, or a local Ollama/TEI server for
// zero-quota iteration. The API key may be empty for unauthenticated local
// servers.
type OpenAICompatible struct {
	APIKey    string
	Model     string
	Endpoint  string
	BatchSize int
	Client    *http.Client
}

type openAIRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

// Embed implements Embedder.
func (o *OpenAICompatible) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	client := o.Client
	if client == nil {
		client = &http.Client{Timeout: httpTimeout}
	}
	return batched(ctx, texts, o.BatchSize, func(ctx context.Context, batch []string) ([][]float32, error) {
		return postEmbeddings(ctx, client, o.Endpoint, o.APIKey, openAIRequest{
			Input: batch,
			Model: o.Model,
		})
	})
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
