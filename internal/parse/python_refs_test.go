package parse

import "testing"

func TestParsePythonRefs(t *testing.T) {
	src := []byte(`import os
from typing import List, Optional as Opt
from .util import helper

class Foo(Base, mixins.Mixed):
    def run(self):
        helper()
        os.path.join('a', 'b')
`)
	refs, err := ParsePythonRefs(src)
	if err != nil {
		t.Fatal(err)
	}

	want := map[string]RefKind{
		"os":       RefImport,
		"List":     RefImport,
		"Optional": RefImport,
		"helper":   RefCall, // also import; both should appear
		"Base":     RefSubclass,
		"Mixed":    RefSubclass,
		"join":     RefCall,
	}
	seen := map[string]map[RefKind]bool{}
	for _, r := range refs {
		if seen[r.Name] == nil {
			seen[r.Name] = map[RefKind]bool{}
		}
		seen[r.Name][r.Kind] = true
	}
	for name, kind := range want {
		if !seen[name][kind] {
			t.Errorf("missing ref %s/%s; got %v", name, kind, refs)
		}
	}
	// `helper` is imported and called — both refs must appear.
	if !seen["helper"][RefImport] {
		t.Errorf("missing import ref for helper")
	}
}
