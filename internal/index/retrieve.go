package index

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/islamborghini/cogni2/internal/retrieve"
)

// Store is the concrete retrieve.Retriever for Stage 1.
var _ retrieve.Retriever = (*Store)(nil)

// Retrieve implements retrieve.Retriever: it embeds the query and returns the k
// chunks with the highest cosine similarity, highest Score first. Scoring is
// exact brute-force over every indexed vector.
func (s *Store) Retrieve(ctx context.Context, query string, k int) ([]retrieve.RetrievedChunk, error) {
	if k <= 0 {
		return nil, nil
	}
	qv, err := s.cfg.QueryEmbedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(qv) != 1 {
		return nil, fmt.Errorf("index: query embedder returned %d vectors", len(qv))
	}
	q := qv[0]

	rows, err := s.loadAll()
	if err != nil {
		return nil, err
	}

	scored := make([]retrieve.RetrievedChunk, len(rows))
	for i, r := range rows {
		scored[i] = retrieve.RetrievedChunk{
			Path:      r.path,
			StartLine: r.start,
			EndLine:   r.end,
			Kind:      r.kind,
			Content:   r.content,
			Score:     cosine(q, r.vec),
		}
	}
	sort.SliceStable(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })

	if k > len(scored) {
		k = len(scored)
	}
	return scored[:k], nil
}

// loadAll lazily loads and caches every chunk's decoded vector. The cache is
// reused across queries and dropped on any write, so a static corpus is read
// from SQLite only once.
func (s *Store) loadAll() ([]vecRow, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cache != nil {
		return s.cache, nil
	}

	rows, err := s.db.Query(`SELECT path, start_line, end_line, kind, content, vec FROM chunks`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var out []vecRow
	for rows.Next() {
		var r vecRow
		var blob []byte
		if err := rows.Scan(&r.path, &r.start, &r.end, &r.kind, &r.content, &blob); err != nil {
			return nil, err
		}
		r.vec = decodeVec(blob)
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	s.cache = out
	return out, nil
}

// cosine returns the cosine similarity of a and b, or 0 if either is zero or
// their dimensions differ.
func cosine(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		av, bv := float64(a[i]), float64(b[i])
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}

// ChangeHandler returns an index.Watcher OnChange callback: changed Python files
// are re-chunked and upserted; deleted ones are dropped. Errors are swallowed
// because the watcher callback cannot return them; the next full BuildAll
// reconciles any missed update.
func (s *Store) ChangeHandler(ctx context.Context, root string) func([]string) {
	return func(paths []string) {
		for _, p := range paths {
			if !IsPython(p) {
				continue
			}
			rel, err := filepath.Rel(root, p)
			if err != nil {
				continue
			}
			if _, err := os.Stat(p); err != nil {
				_ = s.DeleteFile(rel)
				continue
			}
			_ = s.UpsertFile(ctx, root, rel)
		}
	}
}
