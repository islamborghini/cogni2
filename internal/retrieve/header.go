package retrieve

import "fmt"

// sprintHeader renders the visible re-Read anchor for a chunk span.
func sprintHeader(path string, start, end int) string {
	return fmt.Sprintf("%s:%d-%d", path, start, end)
}
