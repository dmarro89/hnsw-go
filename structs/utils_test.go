package structs

import (
	"math"
	"testing"
)

func TestEncodeDecodeHeapItem(t *testing.T) {
	testCases := []struct {
		name     string
		distance float32
		id       int
	}{
		{"Zero values", 0.0, 0},
		{"Positive distance", 1.234, 42},
		{"Negative distance", -1.234, 42},
		{"Very small positive", 1e-10, 1000},
		{"Very small negative", -1e-10, 1000},
		{"Very large positive", 1e10, 999999},
		{"Very large negative", -1e10, 999999},
		{"Max float32", math.MaxFloat32, math.MaxInt32},
		{"Min float32", -math.MaxFloat32, math.MaxInt32},
		{"Special: infinity", float32(math.Inf(1)), 42},
		{"Special: negative infinity", float32(math.Inf(-1)), 42},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Encode
			encoded := EncodeHeapItem(tc.distance, tc.id)

			// Decode
			decodedDist, decodedID := DecodeHeapItem(encoded)

			// Check ID
			if decodedID != tc.id {
				t.Errorf("ID mismatch: got %d, want %d", decodedID, tc.id)
			}

			// Check distance
			if tc.distance != decodedDist {
				t.Errorf("Distance mismatch: got %f, want %f", decodedDist, tc.distance)
			}
		})
	}
}

func TestHeapItemOrdering(t *testing.T) {
	testCases := []struct {
		name      string
		dist1     float32
		id1       int
		dist2     float32
		id2       int
		dist1Less bool
	}{
		{"Different distances", 1.0, 1, 2.0, 1, true},
		{"Different distances negative", -2.0, 1, -1.0, 1, true},
		{"Mixed signs", -1.0, 1, 1.0, 1, true},
		{"Very close values", 1.0000001, 1, 1.0000002, 1, true},
		{"Zero and positive", 0.0, 1, 1.0, 1, true},
		{"Zero and negative", 0.0, 1, -1.0, 1, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			encoded1 := EncodeHeapItem(tc.dist1, tc.id1)
			encoded2 := EncodeHeapItem(tc.dist2, tc.id2)

			if tc.dist1Less {
				if encoded1 >= encoded2 {
					t.Errorf("Expected %f < %f, but encoding gave %d >= %d",
						tc.dist1, tc.dist2, encoded1, encoded2)
				}
			} else {
				if encoded1 < encoded2 {
					t.Errorf("Expected %f >= %f, but encoding gave %d < %d",
						tc.dist1, tc.dist2, encoded1, encoded2)
				}
			}
		})
	}
}

// Aggiungiamo un test separato per distanze uguali
func TestHeapItemOrderingEqualDistances(t *testing.T) {
	dist := float32(1.0)
	id1 := 1
	id2 := 2

	encoded1 := EncodeHeapItem(dist, id1)
	encoded2 := EncodeHeapItem(dist, id2)

	// Per distanze uguali, l'ordine dovrebbe essere determinato dagli ID
	if encoded1 == encoded2 {
		t.Errorf("Equal distances with different IDs should have different encodings")
	}

	// Verifichiamo che decodificando otteniamo le stesse distanze
	decodedDist1, decodedID1 := DecodeHeapItem(encoded1)
	decodedDist2, decodedID2 := DecodeHeapItem(encoded2)

	if decodedDist1 != decodedDist2 {
		t.Errorf("Decoded distances should be equal: got %f and %f", decodedDist1, decodedDist2)
	}

	if decodedID1 != id1 || decodedID2 != id2 {
		t.Errorf("Decoded IDs don't match: got %d and %d, want %d and %d",
			decodedID1, decodedID2, id1, id2)
	}
}

func TestHeapItemEdgeCases(t *testing.T) {
	t.Run("ID boundaries", func(t *testing.T) {
		// Test solo ID non negativi fino a MaxInt32
		testIDs := []int{
			0,
			1,
			100,
			math.MaxInt32,
			math.MaxInt32 - 1,
		}
		dist := float32(1.0)

		for _, id := range testIDs {
			encoded := EncodeHeapItem(dist, id)
			decodedDist, decodedID := DecodeHeapItem(encoded)

			if decodedDist != dist {
				t.Errorf("Distance mismatch for ID %d: got %f, want %f",
					id, decodedDist, dist)
			}

			if decodedID != id {
				t.Errorf("ID mismatch: got %d, want %d", decodedID, id)
			}
		}
	})

	t.Run("Invalid IDs should panic", func(t *testing.T) {
		invalidIDs := []int{
			-1,
			math.MinInt32,
			math.MaxInt32 + 1,
		}

		for _, id := range invalidIDs {
			func() {
				defer func() {
					if r := recover(); r == nil {
						t.Errorf("Expected panic for invalid ID: %d", id)
					}
				}()
				EncodeHeapItem(1.0, id)
			}()
		}
	})
	t.Run("Special float values", func(t *testing.T) {
		specialValues := []float32{
			0.0,
			float32(math.NaN()),
			float32(math.Inf(1)),
			float32(math.Inf(-1)),
		}

		for _, dist := range specialValues {
			encoded := EncodeHeapItem(dist, 1)
			decodedDist, decodedID := DecodeHeapItem(encoded)

			if math.IsNaN(float64(dist)) {
				if !math.IsNaN(float64(decodedDist)) {
					t.Errorf("NaN not preserved: got %f", decodedDist)
				}
			} else if decodedDist != dist {
				t.Errorf("Special value not preserved: got %f, want %f",
					decodedDist, dist)
			}

			if decodedID != 1 {
				t.Errorf("ID mismatch with special value: got %d, want 1", decodedID)
			}
		}
	})
}
