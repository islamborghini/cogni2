package bench

import "testing"

// TestSWEBenchTaskSetValid guards the committed long-horizon set: it must parse,
// pass the same frozen-set invariants as tasks.yaml (pinned SHA, unique ids, gold
// spans), and stay non-trivial. Runs in plain `go test ./...` (no eval tag), so a
// malformed regeneration is caught in CI. The generator
// (scripts/build_swebench_tasks.go) is //go:build ignore and not compiled here.
func TestSWEBenchTaskSetValid(t *testing.T) {
	set, err := Load("tasks-swebench.yaml")
	if err != nil {
		t.Fatalf("load tasks-swebench.yaml: %v", err)
	}
	if set.TargetSHA == "" || set.TargetRepo == "" {
		t.Fatal("swe-bench set must pin a target repo + SHA")
	}
	if len(set.Tasks) < 10 {
		t.Fatalf("swe-bench set has only %d tasks; want >= 10 for a usable long-horizon run", len(set.Tasks))
	}
	for _, task := range set.Tasks {
		if task.Bucket != Localization {
			t.Errorf("task %s: want localization bucket, got %q", task.ID, task.Bucket)
		}
		for _, g := range task.Gold {
			if g.Start > g.End {
				t.Errorf("task %s: gold span %s has start %d > end %d", task.ID, g.Path, g.Start, g.End)
			}
		}
	}
}
