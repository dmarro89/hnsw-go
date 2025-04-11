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
	Neighbors [][]*Node
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
func NewNode(id int, vector []float32, level, maxLevel, maxNeighbors int) *Node {
	// Initialize neighbors slices with pre-allocated capacity
	neighbors := make([][]*Node, maxLevel+1)
	for i := range neighbors {
		neighbors[i] = make([]*Node, 0, maxNeighbors)
	}

	return &Node{
		ID:        id,
		Vector:    vector,
		Level:     level,
		Neighbors: neighbors,
	}
}
