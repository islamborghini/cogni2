//go:build ignore

// build_swebench_tasks generates a long-horizon localization task set from
// SWE-bench Lite, in the same frozen schema as bench/tasks.yaml. The frozen 20 are
// curated 3-8 turn queries; real SWE-bench issues drive far longer agent
// trajectories, which is what Stage 3 (history compression) needs to be measurable
// at all (see bench/results/stage3-cache-redesign.md).
//
// Grading is localization (the user's choice): gold = the symbols the merged-PR gold
// patch touches; success = the agent cites them. Each SWE-bench instance is pinned at
// its own base commit, but we index ONE repo at ONE pinned SHA, so gold is re-anchored
// BY SYMBOL — exactly how tasks.yaml was built ("the definition line of the symbol") —
// using parse.ParsePython at the pinned checkout. Instances whose touched symbols do
// not exist at the pinned SHA are dropped. The result is correct at the indexed SHA
// regardless of the instance's original base commit, and costs one embedding pass.
//
// Run (needs network for the HF datasets-server API + a local checkout at -sha):
//
//	go run scripts/build_swebench_tasks.go \
//	  -repo pytest-dev/pytest -sha <commit> -checkout /path/to/pytest \
//	  -out bench/tasks-swebench.yaml
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/islamborghini/cogni2/bench"
	"github.com/islamborghini/cogni2/internal/parse"
	"gopkg.in/yaml.v3"
)

// hfRows is the HF datasets-server JSON response shape we use.
type hfRows struct {
	Rows []struct {
		Row sweInstance `json:"row"`
	} `json:"rows"`
	NumRowsTotal int `json:"num_rows_total"`
}

type sweInstance struct {
	Repo             string `json:"repo"`
	InstanceID       string `json:"instance_id"`
	BaseCommit       string `json:"base_commit"`
	Patch            string `json:"patch"`
	ProblemStatement string `json:"problem_statement"`
}

// hunkSymbol pulls the enclosing def/class name from a unified-diff hunk header's
// trailing context, e.g. "@@ -242,7 +242,7 @@ def _cstack(left, right):" -> _cstack.
var hunkSymbol = regexp.MustCompile(`@@.*@@\s*(?:async\s+)?(?:def|class)\s+(\w+)`)

// plusFile matches the post-image path line, e.g. "+++ b/src/_pytest/python.py".
var plusFile = regexp.MustCompile(`^\+\+\+ b/(.+)$`)

func main() {
	repo := flag.String("repo", "pytest-dev/pytest", "SWE-bench repo (owner/name) to draw tasks from")
	sha := flag.String("sha", "", "pinned commit the corpus is indexed at (the checkout must be at this SHA)")
	checkout := flag.String("checkout", "", "local path to a checkout of -repo at -sha")
	out := flag.String("out", "bench/tasks-swebench.yaml", "output task file")
	split := flag.String("split", "test", "SWE-bench_Lite split")
	maxTasks := flag.Int("max", 0, "cap on number of tasks (0 = all usable)")
	maxQuery := flag.Int("maxquery", 6000, "truncate problem_statement to this many chars")
	flag.Parse()
	if *sha == "" || *checkout == "" {
		fmt.Fprintln(os.Stderr, "need -sha and -checkout (a checkout of -repo at -sha)")
		os.Exit(2)
	}

	instances, err := fetchLite(*split)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fetch:", err)
		os.Exit(1)
	}

	set := bench.TaskSet{TargetRepo: "https://github.com/" + *repo, TargetSHA: *sha}
	kept, dropped := 0, 0
	for _, in := range instances {
		if in.Repo != *repo {
			continue
		}
		gold := goldFromPatch(in.Patch, *checkout)
		if len(gold) == 0 {
			dropped++
			continue
		}
		q := strings.TrimSpace(in.ProblemStatement)
		if len(q) > *maxQuery {
			q = q[:*maxQuery]
		}
		set.Tasks = append(set.Tasks, bench.Task{
			ID:     in.InstanceID,
			Query:  q,
			Bucket: bench.Localization,
			Gold:   gold,
			Source: "swe-bench-lite:" + in.InstanceID,
		})
		kept++
		if *maxTasks > 0 && kept >= *maxTasks {
			break
		}
	}
	if err := set.Validate(); err != nil {
		fmt.Fprintln(os.Stderr, "generated set fails validation:", err)
		os.Exit(1)
	}

	body, err := yaml.Marshal(set)
	if err != nil {
		fmt.Fprintln(os.Stderr, "marshal:", err)
		os.Exit(1)
	}
	header := fmt.Sprintf(""+
		"# Long-horizon localization benchmark, auto-generated from SWE-bench Lite (%s split).\n"+
		"#\n"+
		"# Target: %s @ %s.\n"+
		"# Source: real GitHub issues + merged-PR gold patches (princeton-nlp/SWE-bench_Lite).\n"+
		"# Gold spans are RE-ANCHORED BY SYMBOL against the checkout at this SHA via tree-sitter\n"+
		"# (the def line of each symbol the gold patch touches), so they are valid at the indexed\n"+
		"# SHA regardless of each instance's original base commit. Unlike the hand-verified\n"+
		"# bench/tasks.yaml, these are AUTO-anchored — a documented provenance difference.\n"+
		"# Regenerate with scripts/build_swebench_tasks.go.\n\n",
		*split, set.TargetRepo, set.TargetSHA)
	if err := os.WriteFile(*out, append([]byte(header), body...), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s: %d tasks from %s (dropped %d with no anchorable gold)\n", *out, kept, *repo, dropped)
}

// fetchLite pages through the whole SWE-bench Lite split via the HF datasets-server.
func fetchLite(split string) ([]sweInstance, error) {
	const base = "https://datasets-server.huggingface.co/rows?dataset=princeton-nlp/SWE-bench_Lite&config=default"
	client := &http.Client{Timeout: 60 * time.Second}
	var all []sweInstance
	for offset := 0; ; offset += 100 {
		url := fmt.Sprintf("%s&split=%s&offset=%d&length=100", base, split, offset)
		resp, err := client.Get(url)
		if err != nil {
			return nil, err
		}
		var page hfRows
		err = json.NewDecoder(resp.Body).Decode(&page)
		_ = resp.Body.Close()
		if err != nil {
			return nil, err
		}
		for _, r := range page.Rows {
			all = append(all, r.Row)
		}
		if len(page.Rows) == 0 || offset+100 >= page.NumRowsTotal {
			break
		}
	}
	return all, nil
}

// goldFromPatch extracts (file, enclosing symbol) pairs from a gold patch's hunk
// headers and resolves each to its definition span at the pinned checkout. Hunks
// without a def/class context, files absent at the pinned SHA, and symbols that no
// longer exist are skipped; the surviving spans are deduplicated.
func goldFromPatch(patch, checkout string) []bench.Gold {
	type want struct{ file, sym string }
	var wants []want
	curFile := ""
	for _, line := range strings.Split(patch, "\n") {
		if m := plusFile.FindStringSubmatch(line); m != nil {
			curFile = m[1]
			continue
		}
		if m := hunkSymbol.FindStringSubmatch(line); m != nil && curFile != "" && curFile != "dev/null" {
			wants = append(wants, want{curFile, m[1]})
		}
	}

	seen := map[string]bool{}
	var gold []bench.Gold
	for _, w := range wants {
		src, err := os.ReadFile(filepath.Join(checkout, w.file))
		if err != nil {
			continue
		}
		syms, err := parse.ParsePython(src)
		if err != nil {
			continue
		}
		for _, s := range syms {
			if s.Name != w.sym {
				continue
			}
			key := fmt.Sprintf("%s:%d-%d", w.file, s.StartLine, s.EndLine)
			if seen[key] {
				continue
			}
			seen[key] = true
			gold = append(gold, bench.Gold{Path: w.file, Start: s.StartLine, End: s.EndLine})
		}
	}
	return gold
}
