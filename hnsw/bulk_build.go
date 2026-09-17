package hnsw

import (
	"errors"
	"runtime"
	"sync"

	"dmarro89.github.com/hnsw-go/structs"
)

// BulkBuildConfig configures the dedicated parallel bulk builder.
type BulkBuildConfig struct {
	Workers        int
	BatchSize      int
	EfConstruction int
}

// DefaultBulkBuildConfig returns a conservative default for parallel bulk builds.
func DefaultBulkBuildConfig() BulkBuildConfig {
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	return BulkBuildConfig{
		Workers:   workers,
		BatchSize: max(32, workers*16),
	}
}

type insertionProposal struct {
	id        int
	level     int
	neighbors [][]int
}

type proposalJob struct {
	idx           int
	snapshotNodes []*structs.Node
	snapshotEntry *structs.Node
	vector        []float32
	id            int
	level         int
}

type proposalResult struct {
	idx      int
	proposal insertionProposal
}

// proposalWorkerPool owns the per-worker search scratch for the whole build.
// Keeping workers alive across batches prevents rebuilding visited-ID arrays
// proportional to the current graph size for every batch.
type proposalWorkerPool struct {
	jobs    chan proposalJob
	results chan proposalResult
	wg      sync.WaitGroup
}

func newProposalWorkerPool(h *HNSW, workers, batchSize, efConstruction, initialVisited int) *proposalWorkerPool {
	pool := &proposalWorkerPool{
		jobs:    make(chan proposalJob, batchSize),
		results: make(chan proposalResult, batchSize),
	}

	for worker := 0; worker < workers; worker++ {
		pool.wg.Add(1)
		go func() {
			defer pool.wg.Done()

			ctx := newSearchContext(max(initialVisited, efConstruction))
			searchResultsBuf := make([]int, 0, efConstruction)
			entryPointsBuf := make([]int, 0, efConstruction)

			for job := range pool.jobs {
				proposal, nextResultsBuf, nextEntryPointsBuf := h.computeInsertionProposal(
					job.snapshotNodes,
					job.snapshotEntry,
					job.vector,
					job.id,
					job.level,
					efConstruction,
					ctx,
					searchResultsBuf,
					entryPointsBuf,
				)
				searchResultsBuf = nextResultsBuf
				entryPointsBuf = nextEntryPointsBuf
				pool.results <- proposalResult{idx: job.idx, proposal: proposal}
			}
		}()
	}

	return pool
}

func (p *proposalWorkerPool) compute(
	snapshotNodes []*structs.Node,
	snapshotEntry *structs.Node,
	vectors [][]float32,
	levels []int,
	startID int,
) []insertionProposal {
	proposals := make([]insertionProposal, len(vectors))

	for i := range vectors {
		p.jobs <- proposalJob{
			idx:           i,
			snapshotNodes: snapshotNodes,
			snapshotEntry: snapshotEntry,
			vector:        vectors[i],
			id:            startID + i,
			level:         levels[i],
		}
	}

	for range vectors {
		result := <-p.results
		proposals[result.idx] = result.proposal
	}

	return proposals
}

func (p *proposalWorkerPool) close() {
	close(p.jobs)
	p.wg.Wait()
}

// BuildParallel appends vectors using a dedicated parallel bulk-construction path.
//
// The expensive search phase runs in parallel on a read-only snapshot of the graph.
// Applying the proposed connections remains serial and deterministic, preserving
// graph invariants while still reducing total build time on multicore machines.
func (h *HNSW) BuildParallel(vectors [][]float32, cfg BulkBuildConfig) error {
	if len(vectors) == 0 {
		return nil
	}

	if cfg.Workers <= 0 {
		cfg.Workers = runtime.GOMAXPROCS(0)
		if cfg.Workers < 1 {
			cfg.Workers = 1
		}
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = max(32, cfg.Workers*16)
	}
	if cfg.EfConstruction <= 0 {
		cfg.EfConstruction = h.EfConstruction
	}
	if cfg.EfConstruction <= 0 {
		return errors.New("EfConstruction must be positive")
	}

	for i, vector := range vectors {
		if len(vector) == 0 {
			return errors.New("vector cannot be empty at batch index " + itoa(i))
		}
	}

	h.mutex.Lock()
	defer h.mutex.Unlock()

	if cfg.BatchSize == 1 {
		oldEfConstruction := h.EfConstruction
		h.EfConstruction = cfg.EfConstruction
		defer func() {
			h.EfConstruction = oldEfConstruction
		}()

		nextID := len(h.Nodes)
		h.ensureNodeCapacity(nextID + len(vectors))
		h.ensureVisitedCapacity(nextID + len(vectors) - 1)
		searchResultsBuf := make([]int, 0, h.EfConstruction)
		for _, vector := range vectors {
			searchResultsBuf = h.insertLocked(vector, nextID, searchResultsBuf)
			nextID++
		}
		return nil
	}

	nextID := len(h.Nodes)
	h.ensureNodeCapacity(nextID + len(vectors))
	h.ensureVisitedCapacity(nextID + len(vectors) - 1)

	levels := make([]int, len(vectors))
	for i := range vectors {
		levels[i] = h.RandomLevel()
	}

	start := 0
	previousScratch := make([]int, 0, max(h.Mmax, h.Mmax0)+1)
	if h.EntryPoint == nil {
		first := insertionProposal{
			id:        nextID,
			level:     levels[0],
			neighbors: make([][]int, levels[0]+1),
		}
		previousScratch = h.applyInsertionProposalScratch(vectors[0], first, previousScratch)
		nextID++
		start = 1
	}

	if start >= len(vectors) {
		return nil
	}

	workerCount := min(cfg.Workers, min(cfg.BatchSize, len(vectors)-start))
	workers := newProposalWorkerPool(
		h,
		workerCount,
		cfg.BatchSize,
		cfg.EfConstruction,
		max(len(h.Nodes), cfg.EfConstruction),
	)
	defer workers.close()

	for batchStart := start; batchStart < len(vectors); batchStart += cfg.BatchSize {
		batchEnd := min(batchStart+cfg.BatchSize, len(vectors))
		snapshotNodes := h.Nodes
		snapshotEntry := h.EntryPoint

		batchVectors := vectors[batchStart:batchEnd]
		batchLevels := levels[batchStart:batchEnd]
		proposals := workers.compute(snapshotNodes, snapshotEntry, batchVectors, batchLevels, nextID)
		for i, proposal := range proposals {
			previousScratch = h.applyInsertionProposalScratch(batchVectors[i], proposal, previousScratch)
		}
		nextID += len(batchVectors)
	}

	return nil
}

func (h *HNSW) computeInsertionProposal(
	snapshotNodes []*structs.Node,
	snapshotEntry *structs.Node,
	vector []float32,
	id, level, efConstruction int,
	ctx *searchContext,
	searchResultsBuf, entryPointsBuf []int,
) (insertionProposal, []int, []int) {
	proposal := insertionProposal{
		id:        id,
		level:     level,
		neighbors: make([][]int, level+1),
	}

	if snapshotEntry == nil {
		return proposal, searchResultsBuf[:0], entryPointsBuf[:0]
	}

	entry := snapshotEntry
	entryPointsBuf = append(entryPointsBuf[:0], entry.ID)
	topLevel := entry.Level

	for lc := topLevel; lc > level; lc-- {
		newEntry := greedySearchLayerNodes(vector, entry, lc, snapshotNodes, h.DistanceFunc)
		if newEntry == nil {
			break
		}
		entry = newEntry
		entryPointsBuf = append(entryPointsBuf[:0], entry.ID)
	}

	maxLayer := min(topLevel, level)
	for lc := maxLayer; lc >= 0; lc-- {
		nearest := searchLayerWithEntriesBufferContext(ctx, snapshotNodes, h.DistanceFunc, vector, entryPointsBuf, efConstruction, lc, searchResultsBuf)
		neighborCount := min(len(nearest), h.M)
		if neighborCount > 0 {
			proposal.neighbors[lc] = append(proposal.neighbors[lc][:0], nearest[:neighborCount]...)
			entryPointsBuf = append(entryPointsBuf[:0], nearest...)
			searchResultsBuf = nearest[:0]
		} else {
			proposal.neighbors[lc] = proposal.neighbors[lc][:0]
			entryPointsBuf = entryPointsBuf[:0]
			searchResultsBuf = searchResultsBuf[:0]
		}
	}

	return proposal, searchResultsBuf, entryPointsBuf
}

func (h *HNSW) applyInsertionProposal(vector []float32, proposal insertionProposal) {
	newNode := h.appendNode(proposal.id, vector, proposal.level)

	if h.EntryPoint == nil {
		h.EntryPoint = newNode
		return
	}

	for level := min(proposal.level, len(proposal.neighbors)-1); level >= 0; level-- {
		maxConn := h.Mmax
		if level == 0 {
			maxConn = h.Mmax0
		}
		h.updateBidirectionalConnectionsBuilder(newNode, proposal.neighbors[level], level, maxConn)
	}

	if proposal.level > h.EntryPoint.Level {
		h.EntryPoint = newNode
	}
}

func (h *HNSW) updateBidirectionalConnectionsBuilder(q *structs.Node, neighbors []int, level int, maxConn int) {
	q.Neighbors[level] = q.Neighbors[level][:0]

	for _, neighborID := range neighbors {
		neighbor := h.Nodes[neighborID]
		if level >= len(neighbor.Neighbors) {
			continue
		}

		accepted := false
		if len(neighbor.Neighbors[level])+1 <= maxConn {
			currentLen := len(neighbor.Neighbors[level])
			if currentLen < cap(neighbor.Neighbors[level]) {
				neighbor.Neighbors[level] = append(neighbor.Neighbors[level], q.ID)
			} else {
				newNeighbors := make([]int, currentLen+1, currentLen+2)
				copy(newNeighbors, neighbor.Neighbors[level])
				newNeighbors[currentLen] = q.ID
				neighbor.Neighbors[level] = newNeighbors
			}
			accepted = true
		} else {
			previous := append([]int(nil), neighbor.Neighbors[level]...)
			candidates := h.selectClosestNeighborIDs(neighbor, neighbor.Neighbors[level], q.ID, maxConn)
			neighbor.Neighbors[level] = neighbor.Neighbors[level][:len(candidates)]
			copy(neighbor.Neighbors[level], candidates)
			accepted = containsNeighborID(neighbor.Neighbors[level], q.ID)

			for _, droppedID := range previous {
				if containsNeighborID(neighbor.Neighbors[level], droppedID) {
					continue
				}
				dropped := h.Nodes[droppedID]
				if level >= len(dropped.Neighbors) {
					continue
				}
				dropped.Neighbors[level] = removeNeighborID(dropped.Neighbors[level], neighbor.ID)
			}
		}

		if accepted {
			q.Neighbors[level] = append(q.Neighbors[level], neighborID)
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
