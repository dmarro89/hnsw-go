package hnsw

import (
	"math"
	"testing"

	"dmarro89.github.com/hnsw-go/structs"
)

// TestInsertEmptyGraph verifies that inserting into an empty graph
// correctly sets the entry point
func TestInsertEmptyGraph(t *testing.T) {
	config := Config{
		M:              10,
		Mmax:           10,
		Mmax0:          10,
		EfConstruction: 16,
		MaxLevel:       5,
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}
	vector := []float32{1.0, 2.0}

	h.Insert(vector, 0)

	if h.EntryPoint == nil {
		t.Fatal("Entry point should not be nil after insertion")
	}
	if h.EntryPoint.ID != 0 {
		t.Errorf("Expected entry point ID to be 0, got %d", h.EntryPoint.ID)
	}
	if len(h.Nodes) != 1 {
		t.Errorf("Expected nodes count to be 1, got %d", len(h.Nodes))
	}
}

// TestInsertPanicsOnEmptyVector verifies that inserting an empty vector triggers a panic
func TestInsertPanicsOnEmptyVector(t *testing.T) {
	config := Config{
		M:              10,
		Mmax:           10,
		Mmax0:          10,
		EfConstruction: 16,
		MaxLevel:       5,
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Insert with empty vector should panic")
		}
	}()

	h.Insert(nil, 0)
}

// TestInsertMultipleItems verifies that inserting multiple elements
// correctly builds the graph structure
func TestInsertMultipleItems(t *testing.T) {
	config := Config{
		M:              2, // Limit M to simplify testing
		Mmax:           2,
		Mmax0:          2,
		EfConstruction: 16,
		MaxLevel:       5,
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	// Insert three vectors
	h.Insert([]float32{1.0, 0.0}, 0)
	h.Insert([]float32{1.0, 1.0}, 1)
	h.Insert([]float32{0.0, 1.0}, 2)

	if len(h.Nodes) != 3 {
		t.Errorf("Expected nodes count to be 3, got %d", len(h.Nodes))
	}

	// Verify that connections were created at level 0
	for _, node := range h.Nodes {
		if len(node.Neighbors[0]) == 0 {
			t.Errorf("Node %d should have neighbors at level 0", node.ID)
		}

	}
}

// TestBidirectionalConnections verifies that bidirectional connections
// are correctly created
func TestBidirectionalConnections(t *testing.T) {
	config := Config{
		M:              2,
		Mmax:           2,
		Mmax0:          2,
		EfConstruction: 16,
		MaxLevel:       1, // Single level
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	// Create three closely positioned nodes
	h.Insert([]float32{1.0, 1.0}, 0)
	h.Insert([]float32{1.1, 1.1}, 1)
	h.Insert([]float32{1.2, 1.2}, 2)

	// Helper function to check if node 'from' is connected to node 'to'
	hasConnection := func(from, to int) bool {
		for _, neighbor := range h.Nodes[from].Neighbors[0] {
			if neighbor.ID == to {
				return true
			}
		}
		return false
	}

	// Check all possible bidirectional connections
	connections := []struct {
		node1 int
		node2 int
	}{
		{0, 1},
		{1, 0},
		{1, 2},
		{2, 1},
		{0, 2},
		{2, 0},
	}

	for _, conn := range connections {
		if !hasConnection(conn.node1, conn.node2) {
			t.Errorf("Expected node %d to be connected to node %d", conn.node1, conn.node2)
		}
	}

	// Alternative: check bidirectional connections more explicitly
	// Check node 0 <-> node 1 connection
	if !hasConnection(0, 1) || !hasConnection(1, 0) {
		t.Errorf("Expected bidirectional connection between nodes 0 and 1")
	}

	// Check node 1 <-> node 2 connection
	if !hasConnection(1, 2) || !hasConnection(2, 1) {
		t.Errorf("Expected bidirectional connection between nodes 1 and 2")
	}

	// Check node 0 <-> node 2 connection
	if !hasConnection(0, 2) || !hasConnection(2, 0) {
		t.Errorf("Expected bidirectional connection between nodes 0 and 2")
	}
}

// TestMaxConnectionsLimit verifies that the number of connections respects the maximum limit
func TestMaxConnectionsLimit(t *testing.T) {
	config := Config{
		M:              2, // M=2 to limit the number of connections
		Mmax:           3, // Different value to easily distinguish from Mmax0
		Mmax0:          4,
		EfConstruction: 16,
		MaxLevel:       2,
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	// Use a deterministic random function to ensure
	// some nodes have different levels
	currentRandIndex := 0
	randValues := []float64{0.9, 0.7, 0.5, 0.3, 0.1, 0.8, 0.6, 0.4, 0.2, 0.05}
	h.RandFunc = func() float64 {
		val := randValues[currentRandIndex]
		currentRandIndex = (currentRandIndex + 1) % len(randValues)
		return val
	}

	// Insert multiple vectors to create many potential connections
	for i := 0; i < 20; i++ {
		vector := []float32{float32(i) * 0.1, float32(i) * 0.1}
		h.Insert(vector, i)
	}

	// Verify no node exceeds the maximum number of connections
	for _, node := range h.Nodes {
		// Check level 0 connections (should respect Mmax0)
		if len(node.Neighbors[0]) > h.Mmax0 {
			t.Errorf("Node %d has %d connections at level 0, exceeding maximum of %d (Mmax0)",
				node.ID, len(node.Neighbors[0]), h.Mmax0)
		}

		// Check connections at higher levels (should respect Mmax)
		for level := 1; level <= node.Level; level++ {
			if level < len(node.Neighbors) {
				if len(node.Neighbors[level]) > h.Mmax {
					t.Errorf("Node %d has %d connections at level %d, exceeding maximum of %d (Mmax)",
						node.ID, len(node.Neighbors[level]), level, h.Mmax)
				}
			}
		}
	}

	// Also verify that M is respected during construction
	// This is more difficult to test directly, but we can check that
	// no node has significantly more than M connections initially
	// (a reasonable heuristic might be 2*M as an upper bound)
	for _, node := range h.Nodes {
		for level := 0; level <= node.Level && level < len(node.Neighbors); level++ {
			maxAllowedForLevel := h.Mmax
			if level == 0 {
				maxAllowedForLevel = h.Mmax0
			}

			if len(node.Neighbors[level]) > maxAllowedForLevel {
				t.Errorf("Node %d at level %d has %d connections, which exceeds the allowed maximum of %d",
					node.ID, level, len(node.Neighbors[level]), maxAllowedForLevel)
			}
		}
	}
}

// TestEntryPointLevelUpdate verifies that the entry point is updated
// when a node with a higher level is inserted
func TestEntryPointLevelUpdate(t *testing.T) {
	config := Config{
		M:              2,
		Mmax:           2,
		Mmax0:          2,
		EfConstruction: 16,
		MaxLevel:       5,
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	// Simulate predetermined levels for nodes
	levels := []int{2, 1, 4, 3} // The third node will have the highest level

	currentLevel := 0
	h.RandFunc = func() float64 {
		level := levels[currentLevel]
		currentLevel++

		return math.Exp(-float64(level)/h.mL) + 0.00001
	}

	// Insert nodes with different levels
	for i := 0; i < len(levels); i++ {
		h.Insert([]float32{float32(i), float32(i)}, i)

		// Verify that the entry point is always the node with the highest level
		expectedEntryPointID := 0
		maxLevel := levels[0]
		for j := 1; j <= i; j++ {
			if levels[j] > maxLevel {
				maxLevel = levels[j]
				expectedEntryPointID = j
			}
		}

		if h.EntryPoint.ID != expectedEntryPointID {
			t.Errorf("After inserting node %d, expected entry point ID to be %d, got %d",
				i, expectedEntryPointID, h.EntryPoint.ID)
		}
	}
}

// TestUpdateBidirectionalConnections verifies the correct functioning
// of the updateBidirectionalConnections function
func TestUpdateBidirectionalConnections(t *testing.T) {
	config := Config{
		M:              2,
		Mmax:           2,
		Mmax0:          3,
		EfConstruction: 16,
		MaxLevel:       1,
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}
	level := 0
	maxConn := 2

	// Manually create nodes
	q := structs.NewNode(0, []float32{0.0, 0.0}, 0, 1, maxConn)
	n1 := structs.NewNode(1, []float32{0.1, 0.0}, 0, 1, maxConn)
	n2 := structs.NewNode(2, []float32{0.2, 0.0}, 0, 1, maxConn)

	// Initialize neighbors of n1 and n2
	n1.Neighbors[level] = []*structs.Node{}
	n2.Neighbors[level] = []*structs.Node{}

	// Update bidirectional connections
	h.updateBidirectionalConnections(q, []*structs.Node{n1, n2}, level, maxConn)

	// Verify that q is connected to n1 and n2
	if len(q.Neighbors[level]) != 2 {
		t.Errorf("Node q should have 2 neighbors, got %d", len(q.Neighbors[level]))
	}

	// Verify that n1 and n2 are connected to q
	foundInN1 := false
	for _, node := range n1.Neighbors[level] {
		if node.ID == q.ID {
			foundInN1 = true
			break
		}
	}

	foundInN2 := false
	for _, node := range n2.Neighbors[level] {
		if node.ID == q.ID {
			foundInN2 = true
			break
		}
	}

	if !foundInN1 || !foundInN2 {
		t.Errorf("Expected bidirectional connections between q and both n1 and n2")
	}
}
