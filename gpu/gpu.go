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
	"sync"
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

	hostBuffer := make([]KeyAddressData, count)
	keys := make([]*btcecv2.PrivateKey, count)
	addrs := make([]string, count)
	types := make([]generator.AddressType, count)

	if err := g.generateCombined(hostBuffer, count); err != nil {
		return nil, nil, nil, err
	}

	resultChan := make(chan result, constants.ChannelBuffer)
	var wg sync.WaitGroup
	var legacyCount, segwitCount, nativeCount uint64

	chunkSize := count / constants.NumWorkers
	if chunkSize < 1000 {
		chunkSize = 1000
	}

	for start := 0; start < count; start += chunkSize {
		wg.Add(1)
		end := start + chunkSize
		if end > count {
			end = count
		}

		go func(start, end int) {
			defer wg.Done()
			localLegacy := uint64(0)
			localSegwit := uint64(0)
			localNative := uint64(0)

			for i := start; i < end; i++ {
				addr, addrType, err := createAddressFromHash160(
					hostBuffer[i].Hash160[:],
					hostBuffer[i].AddressType,
				)
				if err != nil {
					continue
				}
				privKey, _ := btcecv2.PrivKeyFromBytes(hostBuffer[i].PrivateKey[:])

				switch hostBuffer[i].AddressType {
				case 0:
					localLegacy++
				case 1:
					localSegwit++
				case 2:
					localNative++
				}

				select {
				case resultChan <- result{i, privKey, addr, addrType}:
				default:
					constants.TotalDroppedAddresses.Add(1)
				}
			}

			atomic.AddUint64(&legacyCount, localLegacy)
			atomic.AddUint64(&segwitCount, localSegwit)
			atomic.AddUint64(&nativeCount, localNative)
		}(start, end)
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	for r := range resultChan {
		keys[r.idx] = r.privKey
		addrs[r.idx] = r.addr
		types[r.idx] = r.addrType
	}

	constants.GlobalStats.Lock()
	constants.GlobalStats.LegacyCount += int64(legacyCount)
	constants.GlobalStats.SegwitCount += int64(segwitCount)
	constants.GlobalStats.NativeCount += int64(nativeCount)
	constants.GlobalStats.Generated += legacyCount + segwitCount + nativeCount
	constants.GlobalStats.Unlock()

	return keys, addrs, types, nil
}

func NewCUDAGenerator(config *generator.Config) (*CUDAGenerator, error) {
	count, err := cuda.GetDeviceCount()
	if err != cuda.CudaSuccess {
		return nil, fmt.Errorf("CUDA error: %d", err)
	}

	if count == 0 {
		return nil, fmt.Errorf("No CUDA devices found")
	}

	if err := cuda.SetDevice(0); err != cuda.CudaSuccess {
		if err == cuda.CudaErrorDevicesUnavailable {
			return nil, fmt.Errorf("GPU is currently in use by another application")
		}
		return nil, fmt.Errorf("CUDA error setting device: %d", cuda.GetLastError())
	}

	gpuGen, gpuErr := NewGPUKeyGenerator(constants.GPUBatchBufferSize)
	if gpuErr != nil {
		return nil, fmt.Errorf("failed to initialize GPU generator: %v", gpuErr)
	}

	return &CUDAGenerator{
		pool:   generator.NewRNGPool(constants.RNGPoolSize),
		stats:  new(generator.Stats),
		gpuGen: gpuGen,
		gpuLog: log.New(os.Stdout, "", 0),
	}, nil
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
