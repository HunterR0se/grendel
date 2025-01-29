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
	"math/big"
	"os"
	"strings"
	"sync"
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

	// Pre-allocated buffers that persist between Generate calls
	batchBuffers struct {
		keys     []KeyAddressData
		addrs    []string
		types    []generator.AddressType
		privKeys []*btcecv2.PrivateKey
	}

	// Pools for address encoding/decoding
	base58Pool struct {
		ints    sync.Pool    // Pool of *big.Int
		scratch sync.Pool    // Pool of []byte for encoding/decoding
	}
	addrBuilders sync.Pool  // Pool of strings.Builder for address strings
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
		batchBuffers: struct {
			keys     []KeyAddressData
			addrs    []string
			types    []generator.AddressType
			privKeys []*btcecv2.PrivateKey
		}{
			keys:     make([]KeyAddressData, 0, constants.GPUBatchBufferSize),
			addrs:    make([]string, 0, constants.GPUBatchBufferSize),
			types:    make([]generator.AddressType, 0, constants.GPUBatchBufferSize),
			privKeys: make([]*btcecv2.PrivateKey, 0, constants.GPUBatchBufferSize),
		},
		base58Pool: struct {
			ints    sync.Pool
			scratch sync.Pool
		}{
			ints: sync.Pool{
				New: func() interface{} {
					return new(big.Int)
				},
			},
			scratch: sync.Pool{
				New: func() interface{} {
					return make([]byte, 32)
				},
			},
		},
		addrBuilders: sync.Pool{
			New: func() interface{} {
				return &strings.Builder{}
			},
		},
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

	// Grow buffers if needed, reuse if possible
	if cap(g.batchBuffers.keys) < count {
		// Allocate with some extra capacity to avoid frequent resizing
		capacity := count + (count / 4)
		g.batchBuffers.keys = make([]KeyAddressData, count, capacity)
		g.batchBuffers.addrs = make([]string, count, capacity)
		g.batchBuffers.types = make([]generator.AddressType, count, capacity)
		g.batchBuffers.privKeys = make([]*btcecv2.PrivateKey, count, capacity)
	} else {
		// Reuse existing buffers, just reslice
		g.batchBuffers.keys = g.batchBuffers.keys[:count]
		g.batchBuffers.addrs = g.batchBuffers.addrs[:count]
		g.batchBuffers.types = g.batchBuffers.types[:count]
		g.batchBuffers.privKeys = g.batchBuffers.privKeys[:count]
	}

	// Generate on GPU using our pre-allocated buffer
	if err := g.generateCombined(g.batchBuffers.keys, count); err != nil {
		return nil, nil, nil, err
	}

	// Process results using our pre-allocated buffers
	var legacyCount, segwitCount, nativeCount int
	for i := 0; i < count; i++ {
		addr, addrType, err := g.createAddressFromHash160(
			g.batchBuffers.keys[i].Hash160[:],
			g.batchBuffers.keys[i].AddressType,
		)
		if err != nil {
			continue
		}

		g.batchBuffers.addrs[i] = addr
		g.batchBuffers.types[i] = addrType

		switch g.batchBuffers.keys[i].AddressType {
		case 0:
			legacyCount++
		case 1:
			segwitCount++
		case 2:
			nativeCount++
		}
	}

	// Update stats once at the end
	constants.GlobalStats.Lock()
	constants.GlobalStats.LegacyCount += int64(legacyCount)
	constants.GlobalStats.SegwitCount += int64(segwitCount)
	constants.GlobalStats.NativeCount += int64(nativeCount)
	constants.GlobalStats.Generated += uint64(legacyCount + segwitCount + nativeCount)
	constants.GlobalStats.Unlock()

	return g.batchBuffers.privKeys, g.batchBuffers.addrs, g.batchBuffers.types, nil
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

	// Clear batch buffers and pools
	g.batchBuffers.keys = nil
	g.batchBuffers.addrs = nil
	g.batchBuffers.types = nil
	g.batchBuffers.privKeys = nil
	g.base58Pool.ints = sync.Pool{}
	g.base58Pool.scratch = sync.Pool{}
	g.addrBuilders = sync.Pool{}

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

func (g *CUDAGenerator) createAddressFromHash160(hash160 []byte, addrType byte) (string, generator.AddressType, error) {
	var addr btcutil.Address
	var err error
	var addressType generator.AddressType

	// Get string builder from pool
	builder := g.addrBuilders.Get().(*strings.Builder)
	builder.Reset()
	defer g.addrBuilders.Put(builder)

	// Create address object
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

	// Use builder to create address string
	builder.WriteString(addr.EncodeAddress())
	result := builder.String()

	constants.IncrementGenerated()
	return result, addressType, nil
}

func (g *CUDAGenerator) TestGeneration() error {
	return RunBenchmark(constants.GPUTestAddresses, g)
}
