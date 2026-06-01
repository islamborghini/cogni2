// Package index turns a repository into a searchable vector store and serves
// retrieval over it. Chunks come from internal/chunk, vectors from an
// embed.Embedder, and storage is a single pure-Go SQLite file: each chunk's
// vector is a float32 BLOB plus its {path, line span, kind, content}.
//
// At Stage 1 scale (one repo, low tens of thousands of chunks) search is exact
// brute-force cosine over every row — no ANN, no separate service — which is
// the right amount of machinery for a recall@10 benchmark.
package index

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/islamborghini/cogni2/internal/chunk"
	"github.com/islamborghini/cogni2/internal/embed"

	_ "modernc.org/sqlite" // pure-Go SQLite driver
)

const schema = `
CREATE TABLE IF NOT EXISTS chunks (
	id          INTEGER PRIMARY KEY AUTOINCREMENT,
	path        TEXT    NOT NULL,
	start_line  INTEGER NOT NULL,
	end_line    INTEGER NOT NULL,
	kind        TEXT    NOT NULL,
	content     TEXT    NOT NULL,
	content_sha TEXT    NOT NULL,
	dim         INTEGER NOT NULL,
	vec         BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
`

// Config wires the collaborators a Store needs. DocEmbedder embeds chunks at
// index time; QueryEmbedder embeds search queries (a separate instance because
// providers like Voyage encode documents and queries differently).
type Config struct {
	DocEmbedder   embed.Embedder
	QueryEmbedder embed.Embedder
	Tokenizer     chunk.Tokenizer
	ChunkOptions  chunk.Options
}

// Store owns the SQLite handle and an in-memory cache of decoded vectors used by
// retrieval. The cache is invalidated on every write.
type Store struct {
	db  *sql.DB
	cfg Config

	mu    sync.Mutex
	cache []vecRow
}

// vecRow is one chunk loaded into memory for scoring.
type vecRow struct {
	path    string
	start   int
	end     int
	kind    string
	content string
	vec     []float32
}

// Open opens (creating if needed) the SQLite index at dbPath and applies the
// schema. Use ":memory:" for tests. The caller must Close.
func Open(dbPath string, cfg Config) (*Store, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("index: open sqlite: %w", err)
	}
	if _, err := db.Exec(schema); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("index: apply schema: %w", err)
	}
	return &Store{db: db, cfg: cfg}, nil
}

// Close releases the underlying handle.
func (s *Store) Close() error { return s.db.Close() }

// BuildAll indexes every Python file under root and returns the file count.
func (s *Store) BuildAll(ctx context.Context, root string) (int, error) {
	n := 0
	err := walkPython(root, func(rel string) error {
		if err := s.UpsertFile(ctx, root, rel); err != nil {
			return fmt.Errorf("index %s: %w", rel, err)
		}
		n++
		return nil
	})
	return n, err
}

// UpsertFile re-chunks one file and replaces its rows. Chunks whose content is
// unchanged (same SHA already stored for this path) reuse their existing vector,
// so re-indexing an unchanged file costs zero embedding calls.
func (s *Store) UpsertFile(ctx context.Context, root, rel string) error {
	src, err := os.ReadFile(filepath.Join(root, rel))
	if err != nil {
		return err
	}
	chunks, err := chunk.File(rel, src, s.cfg.Tokenizer, s.cfg.ChunkOptions)
	if err != nil {
		return err
	}

	existing, err := s.vectorsByPath(rel)
	if err != nil {
		return err
	}

	shas := make([]string, len(chunks))
	vecs := make([][]float32, len(chunks))
	var toEmbed []string
	var toEmbedIdx []int
	for i, c := range chunks {
		sha := sha256Hex(c.Content)
		shas[i] = sha
		if v, ok := existing[sha]; ok {
			vecs[i] = v
			continue
		}
		toEmbed = append(toEmbed, c.Content)
		toEmbedIdx = append(toEmbedIdx, i)
	}
	if len(toEmbed) > 0 {
		embedded, err := s.cfg.DocEmbedder.Embed(ctx, toEmbed)
		if err != nil {
			return err
		}
		if len(embedded) != len(toEmbed) {
			return fmt.Errorf("index: embedder returned %d vectors for %d inputs", len(embedded), len(toEmbed))
		}
		for j, idx := range toEmbedIdx {
			vecs[idx] = embedded[j]
		}
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM chunks WHERE path = ?`, rel); err != nil {
		_ = tx.Rollback()
		return err
	}
	for i, c := range chunks {
		v := vecs[i]
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO chunks(path, start_line, end_line, kind, content, content_sha, dim, vec)
			 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			c.Path, c.StartLine, c.EndLine, c.Kind, c.Content, shas[i], len(v), encodeVec(v),
		); err != nil {
			_ = tx.Rollback()
			return err
		}
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	s.invalidate()
	return nil
}

// DeleteFile drops all rows for a path (used when a watched file is removed).
func (s *Store) DeleteFile(rel string) error {
	if _, err := s.db.Exec(`DELETE FROM chunks WHERE path = ?`, rel); err != nil {
		return err
	}
	s.invalidate()
	return nil
}

// Count returns the number of indexed chunks.
func (s *Store) Count() (int, error) {
	var n int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM chunks`).Scan(&n)
	return n, err
}

// vectorsByPath returns the stored content_sha → vector map for one path, used
// to skip re-embedding unchanged chunks.
func (s *Store) vectorsByPath(rel string) (map[string][]float32, error) {
	rows, err := s.db.Query(`SELECT content_sha, vec FROM chunks WHERE path = ?`, rel)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	out := map[string][]float32{}
	for rows.Next() {
		var sha string
		var blob []byte
		if err := rows.Scan(&sha, &blob); err != nil {
			return nil, err
		}
		out[sha] = decodeVec(blob)
	}
	return out, rows.Err()
}

func (s *Store) invalidate() {
	s.mu.Lock()
	s.cache = nil
	s.mu.Unlock()
}

// --- helpers ---------------------------------------------------------------

func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func encodeVec(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

func decodeVec(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

// IsPython reports whether path is a Python source file.
func IsPython(path string) bool { return strings.HasSuffix(path, ".py") }

// walkPython calls fn with the repo-relative path of every Python file under
// root, skipping hidden, vendored, and node_modules directories.
func walkPython(root string, fn func(rel string) error) error {
	return filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable entries
		}
		if d.IsDir() {
			base := filepath.Base(path)
			if path != root && (strings.HasPrefix(base, ".") || base == "vendor" || base == "node_modules") {
				return filepath.SkipDir
			}
			return nil
		}
		if !IsPython(path) {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		return fn(rel)
	})
}
