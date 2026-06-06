package bench

import (
	"strings"
	"testing"
)

func TestRenderCompactReport(t *testing.T) {
	set := &TaskSet{TargetRepo: "github.com/django/django", TargetSHA: "abc123"}
	rows := []CompactRow{
		{Budget: 2000, Arm: "rewrite", TreatHitRate: 0.67, NoCacheCostRed: 5.8, GroqCostRed: 0.2, FrontierCostRed: -12.6, Engaged: 9, Tasks: 20},
		{Budget: 2000, Arm: "checkpoint", TreatHitRate: 0.67, NoCacheCostRed: 5.7, GroqCostRed: 0.1, FrontierCostRed: -12.7, Engaged: 9, Tasks: 20},
	}
	horizon := []HorizonRow{
		{Rounds: 4, BaseHitRate: 0.8, RewriteHitRate: 0.7, CheckHitRate: 0.72, RewriteFrontier: -10, CheckFrontier: -8},
		{Rounds: 40, BaseHitRate: 0.85, RewriteHitRate: 0.5, CheckHitRate: 0.78, RewriteFrontier: -20, CheckFrontier: 12},
	}
	md := RenderCompactReport(set, 0.74, rows, horizon, "- replayed 20 trajectories")
	for _, want := range []string{
		"checkpoint compaction", "abc123", "74% cache hit", "frontier 0.1",
		"indistinguishable", "horizon sweep", "model, not a measurement", "SWE-bench",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("compact report missing %q\n%s", want, md)
		}
	}
}

// TestHorizonCrossover is the redesign's load-bearing claim, on a deterministic
// model: as the agent's horizon grows, cache-safe checkpoint compaction pulls ahead
// of the per-turn rewrite policy — strictly higher cache hit rate and lower net cost
// under a frontier discount — because it busts the prefix cache once per checkpoint
// instead of nearly every over-budget turn.
func TestHorizonCrossover(t *testing.T) {
	tok := replayWordTok{}
	agentP := Price{InPer1M: 0.15, OutPer1M: 0.60}
	compP := Price{InPer1M: 0.05, OutPer1M: 0.08}
	rows, err := HorizonSweep(tok, agentP, compP, 200, 300, 60, 12, []int{4, 40})
	if err != nil {
		t.Fatal(err)
	}
	short, long := rows[0], rows[1]

	// Short horizon: the policies are close (the regime the real 20 live in).
	if short.CheckFrontier-short.RewriteFrontier > 5 {
		t.Logf("note: even at 4 rounds checkpoint already leads by %.1f points", short.CheckFrontier-short.RewriteFrontier)
	}
	// Long horizon: checkpoint must lead on both hit rate and net cost.
	if long.CheckHitRate <= long.RewriteHitRate {
		t.Errorf("long horizon: checkpoint hit rate %.2f should exceed rewrite %.2f", long.CheckHitRate, long.RewriteHitRate)
	}
	if long.CheckFrontier <= long.RewriteFrontier {
		t.Errorf("long horizon: checkpoint net cost %.1f%% should beat rewrite %.1f%%", long.CheckFrontier, long.RewriteFrontier)
	}
	// And the gap must widen with horizon (the crossover).
	if (long.CheckFrontier - long.RewriteFrontier) <= (short.CheckFrontier - short.RewriteFrontier) {
		t.Errorf("checkpoint's lead should grow with horizon: short %.1f, long %.1f",
			short.CheckFrontier-short.RewriteFrontier, long.CheckFrontier-long.RewriteFrontier)
	}
}
