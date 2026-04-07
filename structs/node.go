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

// InitNode resets node in place with the specified parameters
func InitNode(node *Node, id int, vector []float32, level, maxLevel, mMax int, mMax0 int) {
	_ = maxLevel

	neighbors := make([][]int, level+1)
	totalCapacity := mMax0
	if level > 0 {
		totalCapacity += level * mMax
	}

	neighborStorage := make([]int, totalCapacity)
	offset := 0

	for i := range neighbors {
		levelCapacity := mMax
		if i == 0 {
			levelCapacity = mMax0
		}
		neighbors[i] = neighborStorage[offset : offset : offset+levelCapacity]
		offset += levelCapacity
	}

	node.ID = id
	node.Vector = vector
	node.Level = level
	node.Neighbors = neighbors
}

// NewNode creates a new Node with the specified parameters
//
// Neighbor storage is backed by a single contiguous slice partitioned per
// level. This reduces the number of heap allocations per node and improves
// cache locality during construction and search
func NewNode(id int, vector []float32, level, maxLevel, mMax int, mMax0 int) *Node {
	node := new(Node)
	InitNode(node, id, vector, level, maxLevel, mMax, mMax0)
	return node
}
