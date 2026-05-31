// Package retrieve defines the locked data contract shared by every stage of
// the context layer, plus the retrieval entry point built in Stage 1.
//
// LOCKED: RetrievedChunk is the spine of the whole project. Stages 2
// (skeleton-first) and 3 (history compression) depend on this exact shape. Do
// not add, remove, or rename fields. If a stage needs more information, carry it
// out of band — never by mutating this struct.
package retrieve

import "context"

// Chunk kinds. A chunk is one syntactically complete unit emitted by the AST
// chunker. class_header chunks carry signatures only (no method bodies).
const (
	KindFunction    = "function"
	KindMethod      = "method"
	KindClassHeader = "class_header"
)

// RetrievedChunk is one unit of code returned by retrieval.
//
// LOCKED SHAPE — see package doc.
type RetrievedChunk struct {
	Path      string  // repo-relative path to the source file
	StartLine int     // 1-based, inclusive
	EndLine   int     // 1-based, inclusive
	Kind      string  // one of Kind* above
	Content   string  // raw source of the chunk
	Score     float32 // cosine similarity to the query; higher is closer
}

// Header returns the "path:startLine-endLine" anchor that every emitted chunk
// carries so the agent can re-Read the full body on demand.
func (c RetrievedChunk) Header() string {
	return sprintHeader(c.Path, c.StartLine, c.EndLine)
}

// Retriever embeds a query and returns the top-k chunks by cosine similarity,
// highest Score first. Implemented in Stage 1.
type Retriever interface {
	Retrieve(ctx context.Context, query string, k int) ([]RetrievedChunk, error)
}
