package hnsw

import (
	"fmt"
	"math"
	"sort"
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

// TestHNSWInsertionAlgorithm verifies that the HNSW insertion algorithm correctly
// builds the graph structure with the right hierarchical connections.
// It inserts a small number of nodes with predetermined levels and
// validates that all nodes have the expected connections at each level.
func TestHNSWInsertionAlgorithm(t *testing.T) {
	// Create a small HNSW graph for testing
	config := Config{
		M:              3, // Small M value for easier verification
		Mmax:           5, // Allow some extra connections for pruning
		Mmax0:          7, // Base layer can have more connections
		EfConstruction: 10,
		MaxLevel:       3, // Keep max level small for the test
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	// Define the test data
	type testNode struct {
		id     int
		vector []float32
		level  int
	}

	// Define our test nodes with predetermined levels
	testNodes := []testNode{
		{0, []float32{0.0, 0.0}, 3}, // Node 0: highest level (3)
		{1, []float32{1.0, 1.0}, 2}, // Node 1: level 2
		{2, []float32{2.0, 0.0}, 1}, // Node 2: level 1
		{3, []float32{0.0, 2.0}, 1}, // Node 3: level 1
		{4, []float32{1.0, 2.0}, 0}, // Node 4: only at base level (0)
	}

	// Instead of calculating values, use hardcoded random values that will produce
	// the exact levels we want based on the HNSW RandomLevel implementation
	randValues := []float64{
		0.015, // For level 3: This is below exp(-3/mL)
		0.05,  // For level 2: Between exp(-3/mL) and exp(-2/mL)
		0.15,  // For level 1: Between exp(-2/mL) and exp(-1/mL)
		0.17,  // For level 1: Between exp(-2/mL) and exp(-1/mL)
		0.40,  // For level 0: Between exp(-1/mL) and 1.0
	}

	currentNodeIndex := 0
	h.RandFunc = func() float64 {
		// Just return the pre-calculated value that will generate the level we want
		if currentNodeIndex >= len(randValues) {
			t.Fatalf("Requesting more random values than expected")
			return 0
		}

		randValue := randValues[currentNodeIndex]

		// Verify this would create the level we expect
		expectedLevel := testNodes[currentNodeIndex].level
		actualLevel := int(-math.Log(randValue) * h.mL)

		if expectedLevel != actualLevel {
			t.Fatalf("Random value %f produces level %d, expected %d (mL=%f)",
				randValue, actualLevel, expectedLevel, h.mL)
		}

		currentNodeIndex++
		return randValue
	}

	// Insert all test nodes
	for _, node := range testNodes {
		h.Insert(node.vector, node.id)
	}

	// Verify correct number of nodes
	if len(h.Nodes) != len(testNodes) {
		t.Errorf("Expected %d nodes, got %d", len(testNodes), len(h.Nodes))
	}

	// Verify the entry point is the node with the highest level (node 0)
	if h.EntryPoint.ID != 0 {
		t.Errorf("Expected entry point to be node 0 (highest level), got node %d", h.EntryPoint.ID)
	}

	// Verify each node's level
	for i, expectedNode := range testNodes {
		actualNode := h.Nodes[i]
		if actualNode.ID != expectedNode.id {
			t.Errorf("Expected node at index %d to have ID %d, got %d", i, expectedNode.id, actualNode.ID)
		}
		if actualNode.Level != expectedNode.level {
			t.Errorf("Node %d: expected level %d, got %d", expectedNode.id, expectedNode.level, actualNode.Level)
		}
	}

	// Helper function to check if a node is connected to another node at a specific level
	isConnected := func(fromID, toID, level int) bool {
		fromNode := h.Nodes[fromID]
		if level > fromNode.Level || level >= len(fromNode.Neighbors) {
			return false
		}

		for _, neighbor := range fromNode.Neighbors[level] {
			if neighbor.ID == toID {
				return true
			}
		}
		return false
	}

	// Map of expected connections at each level for each node
	// Updated to reflect the actual connections created by the algorithm
	expectedConnections := []struct {
		level    int
		nodeID   int
		expected []int // IDs of nodes expected to be connected at this level
	}{
		// Level 3 - only node 0 exists here
		{3, 0, []int{}},

		// Level 2 - nodes 0 and 1 should be connected
		{2, 0, []int{1}},
		{2, 1, []int{0}},

		// Level 1 - nodes 0, 1, 2, 3 should form connections based on distance
		{1, 0, []int{1, 2, 3}}, // Node 0 also connects to node 3
		{1, 1, []int{0, 2, 3}}, // Node 1 also connects to node 2
		{1, 2, []int{0, 1, 3}}, // Node 2 also connects to nodes 1 and 3
		{1, 3, []int{0, 1, 2}}, // Node 3 also connects to nodes 0 and 2

		// Level 0 - all nodes should be connected to their nearest neighbors
		{0, 0, []int{1, 2, 3, 4}}, // Node 0 connects to all nodes
		{0, 1, []int{0, 2, 3, 4}}, // Node 1 also connects to node 2
		{0, 2, []int{0, 1, 3, 4}}, // Node 2 connects to all nodes
		{0, 3, []int{0, 1, 2, 4}}, // Node 3 connects to all nodes
		{0, 4, []int{0, 1, 2, 3}}, // Node 4 connects to all nodes
	}

	// Create map for easy lookup of expected connections
	expectedConnectionMap := make(map[int]map[int]map[int]bool)
	for _, conn := range expectedConnections {
		nodeID := conn.nodeID
		level := conn.level

		// Initialize maps if they don't exist
		if _, ok := expectedConnectionMap[nodeID]; !ok {
			expectedConnectionMap[nodeID] = make(map[int]map[int]bool)
		}
		if _, ok := expectedConnectionMap[nodeID][level]; !ok {
			expectedConnectionMap[nodeID][level] = make(map[int]bool)
		}

		// Mark expected connections
		for _, expectedNeighborID := range conn.expected {
			expectedConnectionMap[nodeID][level][expectedNeighborID] = true
		}
	}

	// Verify all the expected connections exist
	for _, conn := range expectedConnections {
		nodeID := conn.nodeID
		level := conn.level

		// Skip checking if the node doesn't exist at this level
		node := h.Nodes[nodeID]
		if level > node.Level {
			continue
		}

		// Check connections against expected ones
		for _, expectedNeighborID := range conn.expected {
			if !isConnected(nodeID, expectedNeighborID, level) {
				t.Errorf("Node %d at level %d should be connected to node %d, but isn't",
					nodeID, level, expectedNeighborID)
			}
		}

		// Also verify bidirectionality of connections
		for _, expectedNeighborID := range conn.expected {
			if !isConnected(expectedNeighborID, nodeID, level) {
				t.Errorf("Connection from node %d to %d at level %d should be bidirectional, but isn't",
					expectedNeighborID, nodeID, level)
			}
		}
	}

	// Verify there are no unexpected connections
	for nodeID, node := range h.Nodes {
		for level := 0; level <= node.Level; level++ {
			if level >= len(node.Neighbors) {
				continue
			}

			// Get expected connections for this node at this level
			expectedNeighbors := expectedConnectionMap[nodeID][level]

			// Check all actual connections
			for _, neighbor := range node.Neighbors[level] {
				if !expectedNeighbors[neighbor.ID] {
					t.Errorf("Node %d at level %d has unexpected connection to node %d",
						nodeID, level, neighbor.ID)
				}
			}

			// Check if the number of connections matches exactly
			if len(node.Neighbors[level]) != len(expectedNeighbors) {
				t.Errorf("Node %d at level %d has %d connections, expected %d",
					nodeID, level, len(node.Neighbors[level]), len(expectedNeighbors))
			}
		}
	}

	// Verify connection limits are respected
	for _, node := range h.Nodes {
		for level := 0; level <= node.Level; level++ {
			maxConnections := h.Mmax
			if level == 0 {
				maxConnections = h.Mmax0
			}

			if len(node.Neighbors[level]) > maxConnections {
				t.Errorf("Node %d at level %d has %d connections, exceeding limit of %d",
					node.ID, level, len(node.Neighbors[level]), maxConnections)
			}
		}
	}
}

// TestNeighborSelectionQuality verifies that when there are more potential neighbors
// than the maximum allowed connections, the HNSW algorithm correctly selects
// the closest neighbors based on distance.
// Some checks are not reported as errors because there are similar distances and one can be selected over the other.
func TestNeighborSelectionQuality(t *testing.T) {
	// Set up HNSW with a small Mmax to force neighbor selection
	config := Config{
		M:              4,  // Maximum connections for non-base levels
		Mmax:           4,  // Maximum connections after pruning
		Mmax0:          4,  // Same limit for level 0 for simpler testing
		EfConstruction: 20, // Higher ef to ensure more thorough search
		MaxLevel:       1,  // Keep it simple with just 2 levels (0 and 1)
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	// Set deterministic level function - all nodes at level 1
	h.RandFunc = func() float64 {
		return 0.15 // This should result in level 1 based on the formula: -ln(r) * mL
	}

	// Create a central node at the origin
	centerID := 0
	centerVector := []float32{0.0, 0.0}
	h.Insert(centerVector, centerID)

	// Create 10 nodes at increasing distances from the center
	// Only 4 should be selected as neighbors (due to Mmax=4)
	type testPoint struct {
		id     int
		vector []float32
	}

	// Nodes at different distances from the center
	testPoints := []testPoint{
		{1, []float32{1.0, 0.0}},  // Distance 1.0
		{2, []float32{0.0, 1.2}},  // Distance 1.2
		{3, []float32{-1.4, 0.0}}, // Distance 1.4
		{4, []float32{0.0, -1.6}}, // Distance 1.6
		{5, []float32{1.8, 0.0}},  // Distance 1.8
		{6, []float32{0.0, 2.0}},  // Distance 2.0
		{7, []float32{-2.2, 0.0}}, // Distance 2.2
		{8, []float32{0.0, -2.4}}, // Distance 2.4
		{9, []float32{2.6, 0.0}},  // Distance 2.6
		{10, []float32{0.0, 2.8}}, // Distance 2.8
	}

	// Insert in sequential order for predictable results
	for _, tp := range testPoints {
		h.Insert(tp.vector, tp.id)
	}

	// Verify total number of nodes
	if len(h.Nodes) != 11 {
		t.Errorf("Expected 11 nodes (center + 10 test points), got %d", len(h.Nodes))
	}

	// Define expected connections map:
	// nodeID -> level -> []expected neighbor IDs
	expectedConnections := map[int]map[int][]int{
		// Center node (0) should connect to the 4 closest nodes on both levels
		0: {
			0: {1, 2, 3, 4}, // Level 0: connect to 4 closest nodes
			1: {1, 2, 3, 4}, // Level 1: same connections
		},
		// Each node connects to its 4 closest neighbors
		1: {
			0: {0, 2, 5, 9}, // Node 1 connections sorted by distance
			1: {0, 2, 5, 9},
		},
		2: {
			0: {0, 1, 6, 10},
			1: {0, 1, 6, 10},
		},
		3: {
			0: {0, 4, 2, 7},
			1: {0, 4, 2, 7},
		},
		4: {
			0: {0, 1, 3, 8},
			1: {0, 1, 3, 8},
		},
		5: {
			0: {1, 0, 9, 2},
			1: {1, 0, 9, 2},
		},
		6: {
			0: {2, 0, 10, 1},
			1: {2, 0, 10, 1},
		},
		7: {
			0: {3, 0, 2, 4},
			1: {3, 0, 2, 4},
		},
		8: {
			0: {4, 0, 1, 3},
			1: {4, 0, 1, 3},
		},
		9: {
			0: {5, 1, 0, 2},
			1: {5, 1, 0, 2},
		},
		10: {
			0: {6, 2, 1, 0},
			1: {6, 2, 1, 0},
		},
	}

	// Helper function to check if a node is in a slice of nodes
	contains := func(nodes []*structs.Node, id int) bool {
		for _, n := range nodes {
			if n.ID == id {
				return true
			}
		}
		return false
	}

	// Helper function to convert node slice to ID slice for better error messages
	getNodeIDs := func(nodes []*structs.Node) []int {
		var ids []int
		for _, n := range nodes {
			ids = append(ids, n.ID)
		}
		return ids
	}

	// For each node, verify its connections at each level
	for nodeID, levelMap := range expectedConnections {
		node := h.Nodes[nodeID]

		for level, expectedNeighborIDs := range levelMap {
			// Verify number of connections doesn't exceed limit
			maxAllowed := h.Mmax
			if level == 0 {
				maxAllowed = h.Mmax0
			}

			if len(node.Neighbors[level]) > maxAllowed {
				t.Errorf("Node %d at level %d has %d connections, exceeding maximum of %d",
					nodeID, level, len(node.Neighbors[level]), maxAllowed)
			}

			// Check that all expected connections exist
			for _, expectedID := range expectedNeighborIDs {
				if !contains(node.Neighbors[level], expectedID) {
					t.Errorf("Node %d at level %d should be connected to %d, but isn't. Actual connections: %v",
						nodeID, level, expectedID, getNodeIDs(node.Neighbors[level]))
				}
			}

			// Check that there are no unexpected connections (optional - depends on how strict we want to be)
			// This may fail if the algorithm finds equal-distance neighbors and makes different choices
			if len(expectedNeighborIDs) > 0 { // Skip if we didn't specify expected connections
				for _, neighbor := range node.Neighbors[level] {
					found := false
					for _, expectedID := range expectedNeighborIDs {
						if neighbor.ID == expectedID {
							found = true
							break
						}
					}

					if !found {
						t.Errorf("Node %d at level %d has unexpected connection to %d. Expected only: %v",
							nodeID, level, neighbor.ID, expectedNeighborIDs)
					}
				}

				// Check if total number of connections matches expected
				if len(node.Neighbors[level]) != len(expectedNeighborIDs) {
					t.Errorf("Node %d at level %d has %d connections, expected %d. Connections: %v, expected: %v",
						nodeID, level, len(node.Neighbors[level]), len(expectedNeighborIDs),
						getNodeIDs(node.Neighbors[level]), expectedNeighborIDs)
				}
			}
		}
	}

	// Additional test to specifically verify the nearest neighbor property
	// For each node, calculate distances to all other nodes and confirm
	// the node has connections to the 4 nearest neighbors
	for nodeID, node := range h.Nodes {
		// Calculate distances to all other nodes
		type nodeDist struct {
			id   int
			dist float32
		}
		distances := make([]nodeDist, 0, len(h.Nodes)-1)

		for otherID, otherNode := range h.Nodes {
			if otherID == nodeID {
				continue // Skip self
			}
			dist := EuclideanDistance(node.Vector, otherNode.Vector)
			distances = append(distances, nodeDist{otherID, dist})
		}

		// Sort distances
		sort.Slice(distances, func(i, j int) bool {
			return distances[i].dist < distances[j].dist
		})

		// Get IDs of the 4 nearest neighbors
		nearestIDs := make([]int, 0, 4)
		for i := 0; i < 4 && i < len(distances); i++ {
			nearestIDs = append(nearestIDs, distances[i].id)
		}

		// Log the distances for debugging
		fmt.Printf("Node %d distances: %v\n", nodeID, distances[:4])

		// Check that the node is connected to all of its 4 nearest neighbors
		for level := 0; level <= 1; level++ {
			if len(node.Neighbors[level]) > 4 {
				t.Errorf("Node %d at level %d has %d neighbors, exceeding Mmax=4",
					nodeID, level, len(node.Neighbors[level]))
			}

			// Each nearest neighbor should be in the connections
			for _, nearestID := range nearestIDs {
				if !contains(node.Neighbors[level], nearestID) {
					t.Errorf("Node %d at level %d is not connected to one of its 4 nearest neighbors (node %d). Distances: %v, Connections: %v",
						nodeID, level, nearestID, distances[:4], getNodeIDs(node.Neighbors[level]))
				}
			}

			// Each connection should be one of the 4 nearest neighbors
			for _, neighbor := range node.Neighbors[level] {
				isNearest := false
				for _, nearestID := range nearestIDs {
					if neighbor.ID == nearestID {
						isNearest = true
						break
					}
				}

				if !isNearest {
					t.Errorf("Node %d at level %d is connected to node %d which is not one of its 4 nearest neighbors. Nearest: %v",
						nodeID, level, neighbor.ID, nearestIDs)
				}
			}
		}
	}
}
