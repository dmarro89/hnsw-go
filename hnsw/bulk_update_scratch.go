package hnsw

import "dmarro89.github.com/hnsw-go/structs"

// applyInsertionProposalScratch is the parallel bulk-builder apply path with a
// reusable buffer for the pre-prune neighbor snapshot. The scratch is owned by
// BuildParallel's serial apply phase, so no synchronization is required.
func (h *HNSW) applyInsertionProposalScratch(vector []float32, proposal insertionProposal, previousScratch []int) []int {
	newNode := h.appendNode(proposal.id, vector, proposal.level)
	if h.EntryPoint == nil { h.EntryPoint = newNode; return previousScratch }
	for level := min(proposal.level, len(proposal.neighbors)-1); level >= 0; level-- {
		maxConn := h.Mmax; if level == 0 { maxConn = h.Mmax0 }
		previousScratch = h.updateBidirectionalConnectionsBuilderScratch(newNode, proposal.neighbors[level], level, maxConn, previousScratch)
	}
	if proposal.level > h.EntryPoint.Level { h.EntryPoint = newNode }
	return previousScratch
}

func (h *HNSW) updateBidirectionalConnectionsBuilderScratch(q *structs.Node, neighbors []int, level, maxConn int, previousScratch []int) []int {
	q.Neighbors[level] = q.Neighbors[level][:0]
	for _, neighborID := range neighbors {
		neighbor := h.Nodes[neighborID]; if level >= len(neighbor.Neighbors) { continue }
		accepted := false
		if len(neighbor.Neighbors[level])+1 <= maxConn {
			currentLen := len(neighbor.Neighbors[level])
			if currentLen < cap(neighbor.Neighbors[level]) { neighbor.Neighbors[level] = append(neighbor.Neighbors[level], q.ID) } else {
				newNeighbors := make([]int, currentLen+1, currentLen+2); copy(newNeighbors, neighbor.Neighbors[level]); newNeighbors[currentLen] = q.ID; neighbor.Neighbors[level] = newNeighbors
			}
			accepted = true
		} else {
			n := len(neighbor.Neighbors[level]); if cap(previousScratch) < n { previousScratch = make([]int, n) } else { previousScratch = previousScratch[:n] }; copy(previousScratch, neighbor.Neighbors[level])
			candidates := h.selectClosestNeighborIDs(neighbor, neighbor.Neighbors[level], q.ID, maxConn)
			neighbor.Neighbors[level] = neighbor.Neighbors[level][:len(candidates)]; copy(neighbor.Neighbors[level], candidates); accepted = containsNeighborID(neighbor.Neighbors[level], q.ID)
			for _, droppedID := range previousScratch { if containsNeighborID(neighbor.Neighbors[level], droppedID) { continue }; dropped := h.Nodes[droppedID]; if level >= len(dropped.Neighbors) { continue }; dropped.Neighbors[level] = removeNeighborID(dropped.Neighbors[level], neighbor.ID) }
		}
		if accepted { q.Neighbors[level] = append(q.Neighbors[level], neighborID) }
	}
	return previousScratch
}
