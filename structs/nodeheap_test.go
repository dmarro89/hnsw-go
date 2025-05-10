package structs

import (
	"math"
	"testing"
)

func TestNewNodeHeap(t *testing.T) {
	tests := []struct {
		name     string
		distance float32
		id       int
	}{
		{"positive values", 1.5, 42},
		{"zero distance", 0.0, 100},
		{"negative distance", -3.14, 200},
		{"large values", 9999.99, 1000000},
		{"extreme value", math.MaxFloat32, math.MaxInt32},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := NewNodeHeap(tt.distance, tt.id)

			if node == nil {
				t.Fatal("NewNodeHeap returned nil")
			}

			if node.Dist != tt.distance {
				t.Errorf("Distance = %v, want %v", node.Dist, tt.distance)
			}

			if node.Id != tt.id {
				t.Errorf("ID = %v, want %v", node.Id, tt.id)
			}
		})
	}
}
