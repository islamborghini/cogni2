package agent

import "context"

// FakeModel returns scripted responses in order, recording the requests it saw.
// It backs the hermetic loop tests (no network, no key); once the script is
// exhausted it returns a plain assistant message with no tool call, which ends
// the loop.
type FakeModel struct {
	Responses []Response
	Requests  []Request
	n         int
}

// Generate implements Model.
func (f *FakeModel) Generate(_ context.Context, req Request) (Response, error) {
	f.Requests = append(f.Requests, req)
	if f.n >= len(f.Responses) {
		return Response{Message: ChatMessage{Role: RoleAssistant, Content: "(no further action)"}}, nil
	}
	r := f.Responses[f.n]
	f.n++
	return r, nil
}
