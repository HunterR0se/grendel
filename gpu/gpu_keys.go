package gpu

import (
	"Grendel/constants"
	"fmt"
	"unsafe"

	cuda "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

// GPUKeyGenerator handles GPU-accelerated key generation
type GPUKeyGenerator struct {
	devicePrivKeys unsafe.Pointer
	devicePubKeys  unsafe.Pointer
	deviceSeeds    unsafe.Pointer // Add persistent seed storage
	deviceResults  unsafe.Pointer // Add persistent result storage
	batchSize      int
}

func NewGPUKeyGenerator(batchSize int) (*GPUKeyGenerator, error) {
	if batchSize <= 0 {
		batchSize = constants.GPUBatchBufferSize
	}

	// Allocate all memory upfront
	privKeys, err := cuda.Malloc(uint(batchSize * PRIV_KEY_SIZE))
	if err != cuda.CudaSuccess {
		return nil, fmt.Errorf("failed to allocate device memory for private keys: %d", err)
	}

	pubKeys, err := cuda.Malloc(uint(batchSize * PUB_KEY_SIZE))
	if err != cuda.CudaSuccess {
		cuda.Free(privKeys)
		return nil, fmt.Errorf("failed to allocate device memory for public keys: %d", err)
	}

	seeds, err := cuda.Malloc(uint(batchSize * PRIV_KEY_SIZE))
	if err != cuda.CudaSuccess {
		cuda.Free(privKeys)
		cuda.Free(pubKeys)
		return nil, fmt.Errorf("failed to allocate device memory for seeds: %d", err)
	}

	results, err := cuda.Malloc(uint(batchSize * PRIV_KEY_SIZE))
	if err != cuda.CudaSuccess {
		cuda.Free(privKeys)
		cuda.Free(pubKeys)
		cuda.Free(seeds)
		return nil, fmt.Errorf("failed to allocate device memory for results: %d", err)
	}

	return &GPUKeyGenerator{
		devicePrivKeys: privKeys,
		devicePubKeys:  pubKeys,
		deviceSeeds:    seeds,
		deviceResults:  results,
		batchSize:      batchSize,
	}, nil
}

func (g *GPUKeyGenerator) Close() error {
	if g.devicePrivKeys != nil {
		cuda.Free(g.devicePrivKeys)
	}
	if g.devicePubKeys != nil {
		cuda.Free(g.devicePubKeys)
	}
	return nil
}
