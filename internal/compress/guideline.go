package compress

import (
	_ "embed"
	"fmt"
	"os"
	"strings"
)

// defaultGuideline is the compiled-in compression guideline, used by hermetic
// tests and as the fallback when no on-disk override is configured.
//
//go:embed guideline.txt
var defaultGuideline string

// DefaultGuideline returns the embedded guideline text.
func DefaultGuideline() string { return defaultGuideline }

// LoadGuideline returns the guideline to drive compression: the file at path when
// path is non-empty (the editable, no-recompile workflow the refinement script
// targets — point COMPRESS_GUIDELINE at internal/compress/guideline.txt), and the
// embedded default otherwise.
func LoadGuideline(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return defaultGuideline, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("compress: load guideline %q: %w", path, err)
	}
	return string(b), nil
}
