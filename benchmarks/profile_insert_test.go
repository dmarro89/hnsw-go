package benchmarks

import (
	"os"
	"runtime/pprof"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
)

func TestHNSWInsertProfiling(t *testing.T) {
	if testing.Short() {
		t.Skip("Saltando il profiling in modalità short")
	}

	numVectors := 10000
	dimension := 128

	// Genera vettori casuali
	vectors := generateRandomVectors(numVectors, dimension)

	// Crea file di profiling
	cpuFile, err := os.Create("cpu_insert.prof")
	if err != nil {
		t.Fatalf("Impossibile creare file di profilo CPU: %v", err)
	}
	defer cpuFile.Close()

	memFile, err := os.Create("mem_insert.prof")
	if err != nil {
		t.Fatalf("Impossibile creare file di profilo memoria: %v", err)
	}
	defer memFile.Close()

	// Avvia profiling CPU
	if err := pprof.StartCPUProfile(cpuFile); err != nil {
		t.Fatalf("Impossibile avviare profilo CPU: %v", err)
	}
	defer pprof.StopCPUProfile()

	// Inizializza HNSW
	h, err := hnsw.NewHNSW(hnsw.Config{
		M:              16,
		Mmax:           8,
		Mmax0:          16,
		EfConstruction: 200,
		MaxLevel:       16,
		DistanceFunc:   hnsw.EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("Errore nella creazione dell'indice HNSW: %v", err)
	}

	// Esegui inserimenti
	for i := 0; i < numVectors; i++ {
		h.Insert(vectors[i], i)
	}

	// Scrivi profilo memoria
	if err := pprof.WriteHeapProfile(memFile); err != nil {
		t.Fatalf("Impossibile scrivere profilo memoria: %v", err)
	}

	t.Logf("Profili CPU e memoria salvati. Usa 'go tool pprof cpu_insert.prof' e 'go tool pprof mem_insert.prof' per analizzarli")
}
