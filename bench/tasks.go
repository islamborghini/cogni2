// Package bench is the frozen retrieval benchmark: a fixed set of natural-language
// queries over a pinned repository, each with hand-labeled ground-truth code
// spans. Stage 1 scores recall@10 and mean retrieved tokens against it; later
// stages run the identical set so their numbers are comparable.
//
// The loader and metrics here are ordinary (untagged) code so they unit-test
// offline. The end-to-end eval that embeds the real repo lives behind the
// `eval` build tag.
package bench

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Bucket classifies a task by how many gold spans answer it. recall@10 is only
// paper-comparable for localization tasks (|gold| ≤ 10); enumeration tasks are
// bounded by 10/|gold| and reported separately.
type Bucket string

const (
	// Localization: a bounded set of spans fully answers the query.
	Localization Bucket = "localization"
	// Enumeration: the answer is a large set (all subclasses, all callers) that
	// cannot fit in the top-10 by construction.
	Enumeration Bucket = "enumeration"
)

// Gold is one ground-truth relevant span: code at Path on lines [Start,End]
// (1-based, inclusive) that a correct retrieval should surface.
type Gold struct {
	Path  string `yaml:"path"`
	Start int    `yaml:"start"`
	End   int    `yaml:"end"`
}

// Task is one benchmark query plus its labeled answer spans.
type Task struct {
	ID     string `yaml:"id"`
	Query  string `yaml:"query"`
	Bucket Bucket `yaml:"bucket"`
	Gold   []Gold `yaml:"gold"`
	// Source records where the task and its ground truth came from (a v1 task
	// id or a PR), for auditability.
	Source string `yaml:"source,omitempty"`
}

// TaskSet is the YAML root: the pinned target plus the frozen task list.
type TaskSet struct {
	TargetRepo string `yaml:"target_repo"`
	TargetSHA  string `yaml:"target_sha"`
	Tasks      []Task `yaml:"tasks"`
}

// Load reads and validates a task file.
func Load(path string) (*TaskSet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("bench: read tasks: %w", err)
	}
	var set TaskSet
	if err := yaml.Unmarshal(data, &set); err != nil {
		return nil, fmt.Errorf("bench: parse tasks: %w", err)
	}
	if err := set.Validate(); err != nil {
		return nil, err
	}
	return &set, nil
}

// Validate enforces the frozen-set invariants: a pinned SHA, unique non-empty
// ids, a query, a known bucket, and at least one well-formed gold span each.
func (ts *TaskSet) Validate() error {
	if ts.TargetSHA == "" {
		return fmt.Errorf("bench: target_sha is required")
	}
	seen := map[string]bool{}
	for i, t := range ts.Tasks {
		switch {
		case t.ID == "":
			return fmt.Errorf("bench: task[%d]: id is required", i)
		case seen[t.ID]:
			return fmt.Errorf("bench: task %q: duplicate id", t.ID)
		case t.Query == "":
			return fmt.Errorf("bench: task %q: query is required", t.ID)
		case t.Bucket != Localization && t.Bucket != Enumeration:
			return fmt.Errorf("bench: task %q: unknown bucket %q", t.ID, t.Bucket)
		case len(t.Gold) == 0:
			return fmt.Errorf("bench: task %q: at least one gold span is required", t.ID)
		}
		seen[t.ID] = true
		for j, g := range t.Gold {
			if g.Path == "" || g.Start <= 0 || g.End < g.Start {
				return fmt.Errorf("bench: task %q: gold[%d] is malformed: %+v", t.ID, j, g)
			}
		}
	}
	return nil
}
