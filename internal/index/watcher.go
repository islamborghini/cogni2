// Package index wires file changes into the retrieval index. The Watcher is a
// decoupled port of the cogni file watcher: it keeps the fsnotify + debounce
// core but, instead of writing to a symbol store, hands changed paths to an
// OnChange callback. Stage 1 supplies a callback that re-chunks, re-embeds, and
// upserts into the vector store.
package index

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
)

// DefaultDebounce is the window we wait after the last event before flushing.
// Editors often emit several rapid Write/Create events for a single save.
const DefaultDebounce = 200 * time.Millisecond

// Watcher recursively watches a root directory and reports changed files,
// debounced, to OnChange. It is safe to construct, then Run until ctx is done.
type Watcher struct {
	root     string
	debounce time.Duration
	// Match reports whether a path should be tracked (e.g. *.py). If nil, all
	// regular files match.
	Match func(path string) bool
	// OnChange is invoked with the set of changed paths after the debounce
	// window. Required.
	OnChange func(paths []string)

	mu      sync.Mutex
	pending map[string]struct{}
}

// NewWatcher creates a Watcher rooted at dir.
func NewWatcher(dir string) *Watcher {
	return &Watcher{root: dir, debounce: DefaultDebounce, pending: map[string]struct{}{}}
}

// SetDebounce overrides the default debounce window.
func (w *Watcher) SetDebounce(d time.Duration) { w.debounce = d }

func (w *Watcher) matches(path string) bool {
	if w.Match == nil {
		return true
	}
	return w.Match(path)
}

// Run watches until ctx is cancelled. It blocks.
func (w *Watcher) Run(ctx context.Context) error {
	if w.OnChange == nil {
		return errors.New("index: Watcher.OnChange is required")
	}
	fsw, err := fsnotify.NewWatcher()
	if err != nil {
		return err
	}
	defer func() { _ = fsw.Close() }()

	if err := w.addRecursive(fsw, w.root); err != nil {
		return err
	}

	var timer *time.Timer
	var timerC <-chan time.Time
	arm := func() {
		if timer == nil {
			timer = time.NewTimer(w.debounce)
		} else {
			timer.Reset(w.debounce)
		}
		timerC = timer.C
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case ev, ok := <-fsw.Events:
			if !ok {
				return nil
			}
			// New directories must be added to the watch set.
			if ev.Op&fsnotify.Create != 0 {
				if fi, err := os.Stat(ev.Name); err == nil && fi.IsDir() {
					_ = w.addRecursive(fsw, ev.Name)
					continue
				}
			}
			if ev.Op&(fsnotify.Write|fsnotify.Create|fsnotify.Remove|fsnotify.Rename) == 0 {
				continue
			}
			if !w.matches(ev.Name) {
				continue
			}
			w.mu.Lock()
			w.pending[ev.Name] = struct{}{}
			w.mu.Unlock()
			arm()
		case <-timerC:
			w.flush()
			timerC = nil
		case err, ok := <-fsw.Errors:
			if !ok {
				return nil
			}
			if err != nil {
				return err
			}
		}
	}
}

func (w *Watcher) flush() {
	w.mu.Lock()
	if len(w.pending) == 0 {
		w.mu.Unlock()
		return
	}
	paths := make([]string, 0, len(w.pending))
	for p := range w.pending {
		paths = append(paths, p)
	}
	w.pending = map[string]struct{}{}
	w.mu.Unlock()
	w.OnChange(paths)
}

// addRecursive registers dir and all its subdirectories with the fsnotify
// watcher, skipping hidden and vendored directories.
func (w *Watcher) addRecursive(fsw *fsnotify.Watcher, dir string) error {
	return filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable entries
		}
		if !d.IsDir() {
			return nil
		}
		base := filepath.Base(path)
		if path != dir && (strings.HasPrefix(base, ".") || base == "vendor" || base == "node_modules") {
			return filepath.SkipDir
		}
		return fsw.Add(path)
	})
}
