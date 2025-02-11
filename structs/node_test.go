package structs

// func TestNewNode(t *testing.T) {
// 	tests := []struct {
// 		name         string
// 		id           int
// 		vector       []float32
// 		level        int
// 		maxLevel     int
// 		maxNeighbors int
// 	}{
// 		{
// 			name:         "basic node",
// 			id:           1,
// 			vector:       []float32{1.0, 2.0, 3.0},
// 			level:        2,
// 			maxLevel:     5,
// 			maxNeighbors: 10,
// 		},
// 		{
// 			name:         "zero level node",
// 			id:           2,
// 			vector:       []float32{4.0, 5.0},
// 			level:        0,
// 			maxLevel:     3,
// 			maxNeighbors: 5,
// 		},
// 	}

// 	for _, tt := range tests {
// 		t.Run(tt.name, func(t *testing.T) {
// 			node := NewNode(tt.id, tt.vector, tt.level, tt.maxLevel, tt.maxNeighbors)

// 			// Check basic properties
// 			if node.ID != tt.id {
// 				t.Errorf("ID = %v, want %v", node.ID, tt.id)
// 			}
// 			if !reflect.DeepEqual(node.Vector, tt.vector) {
// 				t.Errorf("Vector = %v, want %v", node.Vector, tt.vector)
// 			}
// 			if node.Level != tt.level {
// 				t.Errorf("Level = %v, want %v", node.Level, tt.level)
// 			}

// 			// Check neighbors initialization
// 			if len(node.Neighbors) != tt.maxLevel {
// 				t.Errorf("len(Neighbors) = %v, want %v", len(node.Neighbors), tt.maxLevel)
// 			}
// 			for i, neighbors := range node.Neighbors {
// 				if cap(neighbors) != tt.maxNeighbors {
// 					t.Errorf("cap(Neighbors[%d]) = %v, want %v", i, cap(neighbors), tt.maxNeighbors)
// 				}
// 			}
// 		})
// 	}
// }

// func TestNodeNeighborOperations(t *testing.T) {
// 	tests := []struct {
// 		name string
// 		ops  []struct {
// 			op         string // "add" or "get"
// 			level      int
// 			neighborID int
// 			want       bool  // for add operations
// 			wantIDs    []int // for get operations
// 		}
// 	}{
// 		{
// 			name: "basic operations",
// 			ops: []struct {
// 				op         string
// 				level      int
// 				neighborID int
// 				want       bool
// 				wantIDs    []int
// 			}{
// 				{"add", 0, 2, true, nil},
// 				{"add", 0, 3, true, nil},
// 				{"get", 0, 0, false, []int{2, 3}},
// 				{"add", 1, 4, true, nil},
// 				{"get", 1, 0, false, []int{4}},
// 			},
// 		},
// 		{
// 			name: "invalid operations",
// 			ops: []struct {
// 				op         string
// 				level      int
// 				neighborID int
// 				want       bool
// 				wantIDs    []int
// 			}{
// 				{"add", -1, 2, false, nil},
// 				{"add", 5, 3, false, nil},
// 				{"get", -1, 0, false, nil},
// 				{"get", 5, 0, false, nil},
// 			},
// 		},
// 	}

// 	for _, tt := range tests {
// 		t.Run(tt.name, func(t *testing.T) {
// 			node := NewNode(1, []float32{1.0, 2.0}, 2, 3, 5)

// 			for _, op := range tt.ops {
// 				switch op.op {
// 				case "add":
// 					got := node.AddNeighbor(op.neighborID, op.level, 5)
// 					if got != op.want {
// 						t.Errorf("AddNeighbor(%v, %v) = %v, want %v",
// 							op.neighborID, op.level, got, op.want)
// 					}
// 				case "get":
// 					got := node.GetNeighbors(op.level)
// 					if !reflect.DeepEqual(got, op.wantIDs) {
// 						t.Errorf("GetNeighbors(%v) = %v, want %v",
// 							op.level, got, op.wantIDs)
// 					}
// 				}
// 			}
// 		})
// 	}
// }
