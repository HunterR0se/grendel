package loader

import (
	"Grendel/constants"
	"Grendel/logger"
	"Grendel/parser"
	"Grendel/taproot"
	"encoding/hex"
	"log"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	//"github.com/btcsuite/btcd/btcec"
	"github.com/btcsuite/btcd/chaincfg"
	"github.com/btcsuite/btcd/wire"
	"github.com/btcsuite/btcutil"
	"github.com/shirou/gopsutil/mem"

	// helper functions
	"github.com/btcsuite/btcd/btcec/v2"
	"github.com/btcsuite/btcd/txscript"
)

var loaderLogger *log.Logger

func init() {
	loaderLogger = log.New(os.Stdout, "", log.LstdFlags)
}

type BlockProcessingStatus struct {
	Path      string
	LastTried time.Time
	Attempts  int
}

type BlockLoader struct {
	parser       *parser.Parser // Now properly referenced
	lastScanned  time.Time
	mutex        sync.RWMutex // Better concurrency
	scanTicker   *time.Ticker
	done         chan struct{}
	Stats        *constants.AddressCategory
	failedBlocks map[string]*BlockProcessingStatus
	retryMutex   sync.RWMutex
	stopChan     chan struct{}
	stopOnce     sync.Once
	shutdownChan chan struct{}
}

var (
	magicBuf  = make([]byte, 4)
	sizeBuf   = make([]byte, 4)
	headerBuf = make([]byte, 80) // Block header size
)

var (
	txPool = sync.Pool{
		New: func() interface{} {
			return new(wire.MsgTx)
		},
	}
	addressMap = make(map[string]int64, 1000)
)

func (bl *BlockLoader) Stop() {
	bl.stopOnce.Do(func() {
		close(bl.stopChan)
	})
}

// block helper functions
func IsCoinbaseTx(tx *wire.MsgTx) bool {
	return len(tx.TxIn) == 1 && tx.TxIn[0].PreviousOutPoint.Index == math.MaxUint32
}

func ExtractInputAddresses(script []byte, params *chaincfg.Params) ([]btcutil.Address, error) {
	var addresses []btcutil.Address

	// Try standard script extraction first
	_, addrs, _, err := txscript.ExtractPkScriptAddrs(script, params)
	if err == nil && len(addrs) > 0 {
		return addrs, nil
	}

	// Fallback to manual parsing
	disasm, err := txscript.DisasmString(script)
	if err != nil {
		return nil, err
	}

	parts := strings.Split(disasm, " ")
	for _, part := range parts {
		// Handle public keys
		if len(part) >= 66 {
			data, err := hex.DecodeString(part)
			if err != nil {
				continue
			}

			// Handle different key lengths
			switch len(data) {
			case 33: // Compressed public key
				fallthrough
			case 65: // Uncompressed public key
				pubKey, err := btcec.ParsePubKey(data)
				if err != nil {
					continue
				}

				// Get hash of the public key
				hash := btcutil.Hash160(pubKey.SerializeCompressed())

				// Try all standard address types
				// 1. Legacy address (P2PKH)
				addr, err := btcutil.NewAddressPubKeyHash(hash, params)
				if err == nil {
					addresses = append(addresses, addr)
				}

				// 2. SegWit address (P2WPKH)
				waddr, err := btcutil.NewAddressWitnessPubKeyHash(hash, params)
				if err == nil {
					addresses = append(addresses, waddr)
				}

				// 3. Nested SegWit (P2SH-P2WPKH)
				script := []byte{0x00, 0x14}
				script = append(script, hash...)
				scriptHash := btcutil.Hash160(script)
				naddr, err := btcutil.NewAddressScriptHashFromHash(scriptHash, params)
				if err == nil {
					addresses = append(addresses, naddr)
				}
			}
		}

		// Handle Taproot addresses
		//
		if taproot.IsPayToTaproot(script) {
			logger.LogDebug(loaderLogger, constants.LogDebug, "Attempting to extract Taproot address from script: %x", script)
			taprootAddrs, err := taproot.ExtractTaprootAddresses(script)
			if err != nil {
				logger.LogDebug(loaderLogger, constants.LogDebug, "Failed to extract Taproot address: %v", err)
			} else {
				logger.LogDebug(loaderLogger, constants.LogDebug, "Extracted %d Taproot addresses from script", len(taprootAddrs))
				for _, addr := range taprootAddrs {
					logger.LogDebug(loaderLogger, constants.LogDebug, "Raw address extracted: %s", addr)
					if strings.HasPrefix(addr, "bc1p") {
						addresses = append(addresses, &taproot.TaprootAddress{Addr: addr})
						logger.LogDebug(loaderLogger, constants.LogDebug, "Extracted Taproot address: %s", addr)
					} else {
						logger.LogDebug(loaderLogger, constants.LogDebug, "Ignoring non-Taproot address: %s", addr)
					}
				}
			}
		}

	}

	if len(addresses) > 0 {
		logger.LogDebug(nil, constants.LogDebug,
			"Extracted %d addresses from input script", len(addresses))
	}

	return addresses, nil
}

func ParseComplexScript(script []byte) ([]btcutil.Address, error) {
	var addresses []btcutil.Address

	class := txscript.GetScriptClass(script)
	switch class {
	case txscript.PubKeyHashTy:
		logger.LogDebug(nil, constants.LogDebug, "Found P2PKH script")
	case txscript.ScriptHashTy:
		logger.LogDebug(nil, constants.LogDebug, "Found P2SH script")
	case txscript.WitnessV0PubKeyHashTy:
		logger.LogDebug(nil, constants.LogDebug, "Found P2WPKH script")
	case txscript.WitnessV0ScriptHashTy:
		logger.LogDebug(nil, constants.LogDebug, "Found P2WSH script")
	case txscript.MultiSigTy:
		logger.LogDebug(nil, constants.LogDebug, "Found MultiSig script")
	case txscript.PubKeyTy:
		logger.LogDebug(nil, constants.LogDebug, "Found PubKey script")
	case txscript.NonStandardTy:
		logger.LogDebug(nil, constants.LogDebug, "Found NonStandard script")
	default:
		logger.LogDebug(nil, constants.LogDebug, "Found other type: %v", class)
	}

	// If standard parsing fails, try complex script analysis
	disasm, err := txscript.DisasmString(script)
	if err == nil {
		parts := strings.Split(disasm, " ")
		for _, part := range parts {
			// Look for potential public keys or hashes
			if len(part) >= 40 { // Minimum size for a hex-encoded hash
				data, err := hex.DecodeString(part)
				if err == nil {
					switch len(data) {
					case 20: // Hash160 size
						// Try P2PKH
						if addr, err := btcutil.NewAddressPubKeyHash(data, &chaincfg.MainNetParams); err == nil {
							addresses = append(addresses, addr)
						}
						// Try P2WPKH
						if addr, err := btcutil.NewAddressWitnessPubKeyHash(data, &chaincfg.MainNetParams); err == nil {
							addresses = append(addresses, addr)
						}
					case 32: // SHA256 size
						// Try P2WSH
						if addr, err := btcutil.NewAddressWitnessScriptHash(data, &chaincfg.MainNetParams); err == nil {
							addresses = append(addresses, addr)
						}
					case 33, 65: // Compressed/Uncompressed pubkey sizes
						if pubKey, err := btcec.ParsePubKey(data); err == nil {
							hash := btcutil.Hash160(pubKey.SerializeCompressed())
							if addr, err := btcutil.NewAddressPubKeyHash(hash, &chaincfg.MainNetParams); err == nil {
								addresses = append(addresses, addr)
							}
						}
					}
				}
			}
		}
	}

	if len(addresses) > 0 && constants.DebugMode {
		logger.LogDebug(nil, constants.LogDebug,
			"Found %d addresses in complex script", len(addresses))
	}

	return addresses, nil
}

// general helper functions
func calculateBufferSize() int {
	v, _ := mem.VirtualMemory()
	availableGB := float64(v.Available) / (1024 * 1024 * 1024)
	// Use 25% of available memory for buffer
	bufferSize := int(availableGB * 1024 * 1024 * 1024 / 4)
	if bufferSize < 4*1024*1024 { // Minimum 4MB
		return 4 * 1024 * 1024
	}
	return bufferSize
}

func NewBlockLoader(p *parser.Parser) *BlockLoader {
	if p == nil {
		panic("parser cannot be nil")
	}

	return &BlockLoader{
		parser:       p,
		lastScanned:  time.Now(),
		Stats:        &constants.AddressCategory{},
		scanTicker:   time.NewTicker(30 * time.Second),
		done:         make(chan struct{}),
		failedBlocks: make(map[string]*BlockProcessingStatus),
		stopChan:     make(chan struct{}),
		mutex:        sync.RWMutex{},
		retryMutex:   sync.RWMutex{},
	}
}

// Add this new method
func (bl *BlockLoader) markBlockForRetry(path string) {
	bl.retryMutex.Lock()
	defer bl.retryMutex.Unlock()

	status, exists := bl.failedBlocks[path]
	if !exists {
		bl.failedBlocks[path] = &BlockProcessingStatus{
			Path:      path,
			LastTried: time.Now(),
			Attempts:  1,
		}
	} else {
		status.LastTried = time.Now()
		status.Attempts++
	}
}

func (bl *BlockLoader) shouldRetryBlock(path string) bool {
	bl.retryMutex.RLock()
	defer bl.retryMutex.RUnlock()

	status, exists := bl.failedBlocks[path]
	if !exists {
		return true
	}

	// Exponential backoff: wait longer between each retry attempt
	// Max wait time is ~17 minutes (2^10 seconds)
	waitSeconds := uint64(1 << uint(status.Attempts))
	if waitSeconds > 1024 {
		waitSeconds = 1024
	}

	waitTime := time.Duration(waitSeconds) * time.Second
	return time.Since(status.LastTried) > waitTime
}

func (bl *BlockLoader) Start() {
	// No-op - background processing is handled by startBackgroundLoader
	return
}

// Add after other var declarations
func calculateMaxConcurrent() int {
	v, err := mem.VirtualMemory()
	if err != nil {
		return 2 // Conservative default
	}
	// Use only portion of available memory
	availableGB := float64(v.Available) / (1024 * 1024 * 1024)
	maxConcurrent := int(availableGB / 4) // Each worker needs ~4GB
	if maxConcurrent < 2 {
		maxConcurrent = 2
	}
	if maxConcurrent > runtime.NumCPU() {
		maxConcurrent = runtime.NumCPU()
	}
	return maxConcurrent
}
