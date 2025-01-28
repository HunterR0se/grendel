//go:build !ide && !zed
// +build !ide,!zed

package gpu

/*
#cgo CFLAGS: -I${SRCDIR}/keygen
#cgo LDFLAGS: -L${SRCDIR}/keygen -lkeygen

#include <stdint.h>
#include "keygen.h"
*/
import "C"

import (
	"Grendel/constants"
	"Grendel/generator"
	"fmt"
	"log"
	"os"
	"sync/atomic"
	"unsafe"

	btcecv2 "github.com/btcsuite/btcd/btcec/v2"
	"github.com/btcsuite/btcd/chaincfg"
	"github.com/btcsuite/btcutil"
	cuda "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

type KeyAddressData struct {
	PrivateKey    [32]byte
	PubKeyX       [32]byte
	PubKeyYParity byte
	Hash160       [20]byte
	AddressType   byte
}

type CUDAGenerator struct {
	pool   *generator.RNGPool
	stats  *generator.Stats
	gpuGen *GPUKeyGenerator
	gpuLog *log.Logger
}

type result struct {
	idx      int
	privKey  *btcecv2.PrivateKey
	addr     string
	addrType generator.AddressType
}

func NewCUDAGenerator(config *generator.Config) (*CUDAGenerator, error) {
	// Check for CUDA devices
	count, err := cuda.GetDeviceCount()
	if err != cuda.CudaSuccess {
		return nil, fmt.Errorf("no CUDA devices found")
	}

	if count == 0 {
		return nil, fmt.Errorf("no CUDA devices available")
	}

	// Set the first CUDA device
	err = cuda.SetDevice(0)
	if err != cuda.CudaSuccess {
		if err == cuda.CudaErrorDevicesUnavailable {
			return nil, fmt.Errorf("GPU is currently in use by another application")
		}
		return nil, fmt.Errorf("CUDA error setting device: %d", cuda.GetLastError())
	}

	// Initialize GPU key generator
	gpuGen, gpuErr := NewGPUKeyGenerator(constants.GPUBatchBufferSize)
	if gpuErr != nil {
		return nil, fmt.Errorf("failed to initialize GPU generator: %v", gpuErr)
	}

	// Check for any CUDA errors after initialization
	lastErr := cuda.GetLastError()
	if lastErr != cuda.CudaSuccess {
		return nil, fmt.Errorf("CUDA error: %d", lastErr)
	}

	return &CUDAGenerator{
		pool:   generator.NewRNGPool(constants.RNGPoolSize),
		stats:  new(generator.Stats),
		gpuGen: gpuGen,
		gpuLog: log.New(os.Stdout, "", 0),
	}, nil
}

func (g *CUDAGenerator) Initialize() error {
	if ret := C.test_cuda_device(); ret != 0 {
		return fmt.Errorf("CUDA device test failed")
	}
	return nil
}

func (g *CUDAGenerator) Generate(count int) ([]*btcecv2.PrivateKey, []string, []generator.AddressType, error) {
	if count <= 0 {
		return nil, nil, nil, nil
	}

	if count > g.gpuGen.batchSize {
		count = g.gpuGen.batchSize
	}

	// Pre-allocate all slices
	hostBuffer := make([]KeyAddressData, count)
	addrs := make([]string, count)
	types := make([]generator.AddressType, count)
	// Only allocate private keys when needed
	keys := make([]*btcecv2.PrivateKey, count)

	// Generate on GPU
	if err := g.generateCombined(hostBuffer, count); err != nil {
		return nil, nil, nil, err
	}

	// Process results without creating private keys
	var legacyCount, segwitCount, nativeCount uint64

	// Use a single loop instead of goroutines for this part
	for i := 0; i < count; i++ {
		addr, addrType, err := createAddressFromHash160(
			hostBuffer[i].Hash160[:],
			hostBuffer[i].AddressType,
		)
		if err != nil {
			continue
		}

		addrs[i] = addr
		types[i] = addrType

		// Count stats
		switch hostBuffer[i].AddressType {
		case 0:
			atomic.AddUint64(&legacyCount, 1)
		case 1:
			atomic.AddUint64(&segwitCount, 1)
		case 2:
			atomic.AddUint64(&nativeCount, 1)
		}
	}

	// Update stats once at the end
	constants.GlobalStats.Lock()
	constants.GlobalStats.LegacyCount += int64(legacyCount)
	constants.GlobalStats.SegwitCount += int64(segwitCount)
	constants.GlobalStats.NativeCount += int64(nativeCount)
	constants.GlobalStats.Generated += legacyCount + segwitCount + nativeCount
	constants.GlobalStats.Unlock()

	return keys, addrs, types, nil
}

func (g *CUDAGenerator) generateCombined(keyData []KeyAddressData, count int) error {
	result := C.generate_keys_combined(
		(*C.struct_KeyAddressData)(unsafe.Pointer(&keyData[0])),
		C.int(count),
		C.int(btoi(constants.DebugMode)),
	)
	if result != 0 {
		return fmt.Errorf("combined GPU key generation failed with code: %d", result)
	}
	return nil
}

func (g *CUDAGenerator) Close() error {
	if g.gpuGen != nil {
		if err := g.gpuGen.Close(); err != nil {
			return fmt.Errorf("error closing GPU generator: %v", err)
		}
	}

	if lastErr := cuda.GetLastError(); lastErr != cuda.CudaSuccess {
		return fmt.Errorf("CUDA error on close: %d", lastErr)
	}
	return nil
}

func (g *CUDAGenerator) GetStats() *generator.Stats {
	return g.stats
}

func btoi(b bool) C.int {
	if b {
		return C.int(1)
	}
	return C.int(0)
}

func createAddressFromHash160(hash160 []byte, addrType byte) (string, generator.AddressType, error) {
	var addr btcutil.Address
	var err error
	var addressType generator.AddressType

	switch addrType {
	case 0:
		addr, err = btcutil.NewAddressPubKeyHash(hash160, &chaincfg.MainNetParams)
		addressType = generator.Legacy
	case 1:
		addr, err = btcutil.NewAddressScriptHashFromHash(hash160, &chaincfg.MainNetParams)
		addressType = generator.Segwit
	case 2:
		addr, err = btcutil.NewAddressWitnessPubKeyHash(hash160, &chaincfg.MainNetParams)
		addressType = generator.Native
	}

	if err != nil {
		return "", addressType, err
	}

	constants.IncrementGenerated()
	return addr.EncodeAddress(), addressType, nil
}

func (g *CUDAGenerator) TestGeneration() error {
	return RunBenchmark(constants.GPUTestAddresses, g)
}
