package benchmarks

import (
	"os"
	"testing"
)

func TestHNSWInsertProfiling(t *testing.T) {
	if os.Getenv("HNSW_PROFILE_BUILD") != "1" {
		t.Skip("set HNSW_PROFILE_BUILD=1 to run the serial build profiler")
	}
	runSerialBuildProfile(t)
}
