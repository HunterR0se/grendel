package gpu

import (
	"Grendel/generator"

	btcecv2 "github.com/btcsuite/btcd/btcec/v2"
	cuda "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

const (
	threadsPerBlock = 1024
	maxBlocks       = 65535
	PRIV_KEY_SIZE   = 32
	PUB_KEY_SIZE    = 33
	COMBINED_SIZE   = 65 // New: private key + public key X + parity bit
	addressSize     = 42
)

func NewGenerator(useGPU bool) generator.Generator {
	if useGPU {
		// Check if CUDA is available
		count, err := cuda.GetDeviceCount()
		if err == cuda.CudaSuccess && count > 0 {
			// Use CUDA if available
			if gen, err := NewCUDAGenerator(&generator.Config{}); err == nil {
				return gen
			}
			// If CUDA initialization fails, fall back to CPU
		}
		// If CUDA is not available or initialization fails, use CPU
	}

	// Fallback to CPU
	gen, _ := NewCPUGenerator(&generator.Config{})
	return gen
}

// Generator is an interface that both CPU and OpenCL generators will implement
type Generator interface {
	Generate(count int) ([]*btcecv2.PrivateKey, []string, []generator.AddressType, error)
	Close() error
}
