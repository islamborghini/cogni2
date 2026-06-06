package bench

import (
	"context"
	"strings"
	"testing"

	"github.com/islamborghini/cogni2/internal/agent"
	"github.com/islamborghini/cogni2/internal/compress"
)

// TestCacheBaselineNearTotalHitRate: an append-only (uncompressed) trajectory never
// rewrites an earlier message, so every turn's prefix fully contains the previous
// turn's prompt — the prefix cache should hit on nearly all input, missing only the
// freshly appended tail and the cold first turn.
func TestCacheBaselineNearTotalHitRate(t *testing.T) {
	msgs := buildTrajectory()
	cc, err := ReplayCacheCost(context.Background(), msgs, 2, nil, 0, replayWordTok{})
	if err != nil {
		t.Fatal(err)
	}
	if cc.Cached == 0 {
		t.Fatal("append-only baseline should serve most input from cache")
	}
	if cc.HitRate() < 0.5 {
		t.Errorf("baseline hit rate %.2f unexpectedly low for append-only growth", cc.HitRate())
	}
}

// TestCacheCompressionBustsPrefix: compression rewrites an older observation, which
// moves the first divergence earlier and re-bills the tail after it. So the
// treatment arm must have a STRICTLY LOWER cache hit rate than the baseline on the
// same trajectory — the prompt-cache trap, made visible.
func TestCacheCompressionBustsPrefix(t *testing.T) {
	msgs := buildTrajectory()
	tok := replayWordTok{}

	base, err := ReplayCacheCost(context.Background(), msgs, 2, nil, 0, tok)
	if err != nil {
		t.Fatal(err)
	}
	ms := &agent.MeteringSummarizer{Inner: SizeSummarizer{MaxWords: 4}, Tok: tok}
	comp := &compress.GuidelineCompressor{Summarizer: ms, Tok: tok}
	treat, err := ReplayCacheCost(context.Background(), msgs, 2, comp, 20, tok)
	if err != nil {
		t.Fatal(err)
	}
	if !treat.Engaged {
		t.Fatal("compression should have engaged")
	}
	if treat.HitRate() >= base.HitRate() {
		t.Fatalf("compression should LOWER the cache hit rate: baseline %.2f, treatment %.2f",
			base.HitRate(), treat.HitRate())
	}
}

func TestRenderCacheReport(t *testing.T) {
	set := &TaskSet{TargetRepo: "github.com/django/django", TargetSHA: "abc123"}
	rows := []CacheRow{
		{Budget: 2000, BaseHitRate: 0.74, TreatHitRate: 0.67, NoCacheCostRed: 5.8, GroqCostRed: 0.2, FrontierCostRed: -12.6, Engaged: 9, Tasks: 20},
		{Budget: 500, BaseHitRate: 0.74, TreatHitRate: 0.66, NoCacheCostRed: 4.4, GroqCostRed: -2.5, FrontierCostRed: -17.5, Engaged: 11, Tasks: 20},
	}
	md := RenderCacheReport(set, 0.74, rows, "- replayed 20 trajectories")
	for _, want := range []string{
		"prompt caching", "abc123", "74% cache hit", "frontier 0.1", "+5.8%", "-17.5%", "cache trap",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("cache report missing %q\n%s", want, md)
		}
	}
}

// TestCacheCanReverseRawTokenWin is the headline check: the raw-token replay can
// show compression saving tokens while the cache-aware bill goes the other way,
// because the saved tokens were cheap cache hits and the busted tail is full price.
// At a deep cache discount the treatment's net input cost should be able to exceed
// the baseline's even when its raw token count is lower.
func TestCacheCanReverseRawTokenWin(t *testing.T) {
	msgs := buildTrajectory()
	tok := replayWordTok{}

	// Raw-token view: compression is a win (this is the existing replay's claim).
	unc, _ := ReplayCost(context.Background(), msgs, 2, nil, 0, tok)
	ms := &agent.MeteringSummarizer{Inner: SizeSummarizer{MaxWords: 4}, Tok: tok}
	comp := &compress.GuidelineCompressor{Summarizer: ms, Tok: tok}
	cmp, _ := ReplayCost(context.Background(), msgs, 2, comp, 20, tok)
	if cmp.Input >= unc.Input {
		t.Fatalf("raw replay precondition: compression should cut re-sent tokens, %d vs %d", cmp.Input, unc.Input)
	}

	// Cache-aware view at a frontier-style read discount (0.1): cached tokens are
	// nearly free, so the baseline's huge cache hit beats the treatment's busted tail.
	const mul = 0.1
	price := Price{InPer1M: 1, OutPer1M: 0}
	baseCC, _ := ReplayCacheCost(context.Background(), msgs, 2, nil, 0, tok)
	ms2 := &agent.MeteringSummarizer{Inner: SizeSummarizer{MaxWords: 4}, Tok: tok}
	comp2 := &compress.GuidelineCompressor{Summarizer: ms2, Tok: tok}
	treatCC, _ := ReplayCacheCost(context.Background(), msgs, 2, comp2, 20, tok)

	baseBill := CacheNetCostUSD(baseCC, price, mul, 0, Price{})
	treatBill := CacheNetCostUSD(treatCC, price, mul, ms2.InputTokens+ms2.OutputTokens, Price{InPer1M: 1})
	if treatBill <= baseBill {
		t.Fatalf("cache-aware input bill should be able to exceed baseline under a deep discount: base %.4f, treat %.4f",
			baseBill, treatBill)
	}
}
