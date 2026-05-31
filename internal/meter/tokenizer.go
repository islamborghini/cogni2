package meter

import (
	"sync"

	"github.com/pkoukk/tiktoken-go"
)

// tiktokenTokenizer is a tiktoken-compatible Tokenizer. Anthropic does not
// publish its tokenizer, so we use OpenAI's BPE (o200k_base) as the standard
// tiktoken-compatible proxy. The absolute counts are an estimate; what matters
// for the ablation is that the SAME tokenizer is applied to every bucket in
// every stage, so reductions are comparable.
type tiktokenTokenizer struct {
	enc *tiktoken.Tiktoken
}

var (
	defaultTok  Tokenizer
	defaultOnce sync.Once
	defaultErr  error
)

// Default returns a process-wide tiktoken-compatible tokenizer, initialised
// once. The first call may download/load the BPE vocab.
func Default() (Tokenizer, error) {
	defaultOnce.Do(func() {
		enc, err := tiktoken.GetEncoding("o200k_base")
		if err != nil {
			defaultErr = err
			return
		}
		defaultTok = &tiktokenTokenizer{enc: enc}
	})
	return defaultTok, defaultErr
}

func (t *tiktokenTokenizer) Count(text string) int {
	if text == "" {
		return 0
	}
	return len(t.enc.Encode(text, nil, nil))
}
