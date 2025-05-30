package structs

// Node represents a vector in the HNSW graph. Each node contains a vector of coordinates
// and maintains connections to its neighbors at different levels of the graph.
type Node struct {
	// ID uniquely identifies the node in the graph
	ID int

	// Vector contains the coordinates that represent this node in the space
	Vector []float32

	// Level indicates the highest level where this node appears in the graph
	Level int

	// Neighbors stores the IDs of neighboring nodes for each level
	// The first index represents the level, the second index represents neighbors at that level
	Neighbors [][]int
}

// NewNode creates a new Node with the specified parameters.
// Parameters:
//   - id: unique identifier for the node
//   - vector: coordinates of the node in the space
//   - level: maximum level for this node
//   - maxLevel: maximum number of levels in the graph
//   - maxNeighbors: maximum number of neighbors per level
//
// Returns a pointer to the newly created Node.
func NewNode(id int, vector []float32, level, maxLevel, mMax int, mMax0 int) *Node {
	// Initialize neighbors slices with pre-allocated capacity
	neighbors := make([][]int, level+1)
	for i := range neighbors {
		if i == 0 {
			// Level 0 neighbors are initialized with a capacity of mMax0
			neighbors[i] = make([]int, 0, mMax0)
		} else {
			// All other levels should also have zero initial length but proper capacity
			neighbors[i] = make([]int, 0, mMax)
		}
	}

	return &Node{
		ID:        id,
		Vector:    vector,
		Level:     level,
		Neighbors: neighbors,
	}
}
