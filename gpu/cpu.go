package gpu

import (
	"Grendel/constants"
	"Grendel/generator"

	"sync"

	btcecv2 "github.com/btcsuite/btcd/btcec/v2"
)

type CPUGenerator struct {
	pool  *generator.RNGPool
	stats *generator.Stats
}

// NewCPUGenerator creates a new CPU-based generator for
// Bitcoin addresses with the provided configuration.
//
// It initializes an RNG pool and statistics tracker.
// Returns a CPUGenerator instance and any initialization errors.
func NewCPUGenerator(config *generator.Config) (*CPUGenerator, error) {
	return &CPUGenerator{
		pool:  generator.NewRNGPool(constants.RNGPoolSize),
		stats: new(generator.Stats),
	}, nil
}

func (g *CPUGenerator) GetStats() *generator.Stats {
	return g.stats
}

// // Generate produces multiple Bitcoin addresses and their corresponding private keys.
//
// The count parameter determines how many addresses to generate. The process is
// parallelized across multiple workers for better performance.
//
// Returns:
// - A slice of private keys
// - A slice of address strings
// - A slice of address types indicating format (P2PKH, P2WPKH, etc)
// - Any error that occurred during generation
//
// The returned slices may be shorter than the requested count if some
// generations failed. Failed generations are silently skipped.
func (g *CPUGenerator) Generate(count int) ([]*btcecv2.PrivateKey, []string, []generator.AddressType, error) {
	if count <= 0 {
		return nil, nil, nil, nil
	}

	keys := make([]*btcecv2.PrivateKey, count)
	addrs := make([]string, count)
	types := make([]generator.AddressType, count)

	var mu sync.Mutex
	validCount := 0

	workers := constants.NumWorkers
	batchSize := (count + workers - 1) / workers

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		start := w * batchSize
		end := start + batchSize
		if end > count {
			end = count
		}

		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				seed := g.pool.Get()
				key, addr, addrType, err := generator.GenerateRawPrivateKey(seed, g.stats)
				if err != nil {
					continue
				}

				mu.Lock()
				if validCount < count {
					keys[validCount] = key
					addrs[validCount] = addr
					types[validCount] = addrType
					validCount++
				}
				mu.Unlock()
			}
		}(start, end)
	}

	wg.Wait()

	return keys[:validCount], addrs[:validCount], types[:validCount], nil
}

func (g *CPUGenerator) GenerateBatch(count int) ([]string, error) {
	if count <= 0 {
		return nil, nil
	}
	_, addrs, _, err := g.Generate(count)
	return addrs, err
}

func (g *CPUGenerator) Close() error {
	return nil
}
