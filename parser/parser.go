package parser

import (
	"Grendel/addresses"
	"Grendel/constants"
	"Grendel/logger"
	"Grendel/utils"
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"runtime/pprof"
	"sync"
	"time"

	"github.com/syndtr/goleveldb/leveldb"
	"github.com/syndtr/goleveldb/leveldb/filter"
	"github.com/syndtr/goleveldb/leveldb/opt"
)

type Parser struct {
	DB                 *leveldb.DB
	Logger             *log.Logger
	BlockHeight        int64
	AddressCount       uint64
	DataDir            string
	DBPath             string
	ForceReparse       bool
	cache              map[string]bool
	cacheMux           sync.RWMutex
	addressMutex       sync.RWMutex
	BitcoinDir         string
	processedBlocksMux sync.RWMutex
	Stats              *constants.AddressCategory
	processedBlocks    map[string]bool
	LastBlockHeight    int64
	BlocksProcessed    int64
	AddressesFound     int64
	lastUpdateTime     time.Time
	addresses          map[uint64]uint64 // hash -> balance
}

var (
	EnableProfiling = false   // Global control for profiling
	profileOnce     sync.Once // Ensure we only create one profile
)

func fastHash(s string) uint64 {
	// Use FNV-1a hash for better performance
	const fnvPrime = 1099511628211
	const fnvOffset = 14695981039346656037

	hash := uint64(fnvOffset)
	for i := 0; i < len(s); i++ {
		hash ^= uint64(s[i])
		hash *= fnvPrime
	}
	return hash
}

func (p *Parser) CheckAddressBatch(addresses []string, results []bool, balances []int64) {
	var totalTime, lockTime, lookupTime time.Duration
	var lookupCount, hitCount int

	if EnableProfiling {
		profileOnce.Do(func() {
			f, err := os.Create("cpu_profile.pprof")
			if err != nil {
				log.Fatal(err)
			}
			if err := pprof.StartCPUProfile(f); err != nil {
				log.Fatal(err)
			}
		})
	}

	start := time.Now()
	lockStart := time.Now()
	p.addressMutex.RLock()
	lockTime = time.Since(lockStart)

	// Pre-allocate hash slice
	hashes := make([]uint64, len(addresses))

	lookupStart := time.Now()

	// Calculate all hashes first for better cache locality
	for i, addr := range addresses {
		hashes[i] = fastHash(addr)
	}

	// Process in chunks for better cache utilization
	const chunkSize = 1024
	for i := 0; i < len(addresses); i += chunkSize {
		end := i + chunkSize
		if end > len(addresses) {
			end = len(addresses)
		}

		// Check chunk of addresses
		for j := i; j < end; j++ {
			if EnableProfiling {
				lookupCount++
			}

			if balance, exists := p.addresses[hashes[j]]; exists {
				if EnableProfiling {
					hitCount++
				}
				results[j] = true
				balances[j] = int64(balance)
			}
		}
	}

	p.addressMutex.RUnlock()

	if EnableProfiling {
		lookupTime = time.Since(lookupStart)
		totalTime = time.Since(start)
		p.Logger.Printf("CheckAddressBatch Profile:\n"+
			"Total Time: %v\n"+
			"Lock Time: %v\n"+
			"Lookup Time: %v\n"+
			"Lookups: %d\n"+
			"Hits: %d\n"+
			"Avg Time per Lookup: %v\n"+
			"Memory Map Size: %d\n",
			totalTime,
			lockTime,
			lookupTime,
			lookupCount,
			hitCount,
			lookupTime/time.Duration(lookupCount),
			len(p.addresses))
	}
}

func (p *Parser) LoadAllAddresses() (uint64, error) {
	iter := p.DB.NewIterator(nil, nil)
	defer iter.Release()

	// Pre-allocate main map
	p.addresses = make(map[uint64]uint64, p.AddressCount)

	prefix := []byte("addr:")
	count := uint64(0)

	for iter.Next() {
		key := iter.Key()
		if len(key) > 5 && bytes.Equal(key[:5], prefix) {
			addr := string(key[5:])
			balance := binary.LittleEndian.Uint64(iter.Value())

			// Store with hashed key
			hash := fastHash(addr)
			p.addresses[hash] = balance
			count++
		}
	}

	return count, nil
}

// Update AddTestAddress
func (p *Parser) AddTestAddress(addr string) error {
	p.addressMutex.Lock()
	defer p.addressMutex.Unlock()

	testBalance := uint64(10)
	balanceBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(balanceBytes, testBalance)

	// Add to main map with hash
	hash := fastHash(addr)
	p.addresses[hash] = testBalance

	if err := p.DB.Put([]byte("addr:"+addr), balanceBytes, nil); err != nil {
		return err
	}

	addresses.CategorizeAddress(addr, p.Stats)
	p.AddressCount++
	return nil
}

func (p *Parser) GetAddresses() []string {
	p.addressMutex.RLock()
	addresses := make([]string, 0, len(p.addresses))
	// Just return the hashed keys since that's what we use for matching
	for hash := range p.addresses {
		addresses = append(addresses, fmt.Sprintf("%x", hash))
	}
	p.addressMutex.RUnlock()
	return addresses
}

// -------------------------------------------------------------

func (p *Parser) ResetProcessedBlocks() error {
	p.processedBlocksMux.Lock()
	p.processedBlocks = make(map[string]bool, constants.InitialProcessedBlocksCapacity)
	p.processedBlocksMux.Unlock()
	return nil
}

func (p *Parser) loadStats() error {
	iter := p.DB.NewIterator(nil, nil)
	defer iter.Release()

	prefix := []byte("addr:")
	for iter.Next() {
		key := iter.Key()
		if bytes.HasPrefix(key, prefix) {
			addr := string(key[5:])
			addresses.CategorizeAddress(addr, p.Stats)
		}
	}

	// Update global stats
	constants.GlobalStats.Lock()
	constants.GlobalStats.LegacyCount = int64(p.Stats.LegacyCount)
	constants.GlobalStats.SegwitCount = int64(p.Stats.SegwitCount)
	constants.GlobalStats.NativeCount = int64(p.Stats.NativeCount)
	constants.GlobalStats.TotalCount = constants.GlobalStats.LegacyCount +
		constants.GlobalStats.SegwitCount +
		constants.GlobalStats.NativeCount
	constants.GlobalStats.Unlock()

	return iter.Error()
}

// ------------------------------------------------------------------
func NewParser(localLog *log.Logger,
	bitcoinDir,
	dbPath string,
	forceReparse bool) (*Parser, error) {

	sizeGB := utils.CalculateDBSize(dbPath)
	var action string
	if forceReparse || sizeGB == 0 {
		action = "Creating"
	} else {
		action = "Loading"
	}

	if sizeGB > 0 {
		logger.LogStatus(localLog, constants.LogInfo,
			"%s Addresses: %.2fGB", action, sizeGB)
	}

	// Optimize database options based on available memory
	opts := &opt.Options{
		BlockCacheCapacity:     constants.MinBufferSize,     // Base block cache
		WriteBuffer:            constants.MinBufferSize / 2, // Smaller write buffer
		CompactionTableSize:    constants.MinBufferSize / 4, // Conservative compaction
		OpenFilesCacheCapacity: 1000,                        // Increased from 500
		Filter:                 filter.NewBloomFilter(10),   // Keep bloom filter
		BlockRestartInterval:   16,                          // Increased from 8
		BlockSize:              32 * 1024,                   // Increased block size
		WriteL0SlowdownTrigger: 16,                          // Doubled
		WriteL0PauseTrigger:    48,                          // Doubled
	}

	p := &Parser{
		Logger:          localLog,
		BitcoinDir:      bitcoinDir,
		DBPath:          dbPath,
		ForceReparse:    forceReparse,
		processedBlocks: make(map[string]bool, constants.InitialProcessedBlocksCapacity),
		addresses:       make(map[uint64]uint64, constants.InitialAddressesCapacity),
		cache:           make(map[string]bool, constants.InitialCacheCapacity),
		Stats:           &constants.AddressCategory{},
	}

	startTime := time.Now()
	var err error
	if p.DB, err = leveldb.OpenFile(dbPath, opts); err != nil {
		return nil, fmt.Errorf("failed to open database: %v", err)
	}

	// Only load statistics initially
	if err := p.loadStats(); err != nil {
		return nil, fmt.Errorf("failed to load stats: %v", err)
	}

	if forceReparse {
		if err := p.ResetProcessedBlocks(); err != nil {
			logger.LogError(localLog, constants.LogError, err,
				"Failed to reset processed blocks")
		}
	}

	// Count addresses and update stats without full load
	if count, err := p.CountAddresses(); err != nil {
		return nil, fmt.Errorf("failed to count addresses: %v", err)
	} else {
		elapsed := time.Since(startTime)
		if count > 0 {
			logger.LogStatus(localLog, constants.LogInfo,
				"Total Addresses:   %s (%.1f seconds)",
				utils.FormatWithCommas(int(count)),
				elapsed.Seconds())
		}
		p.AddressCount = count
	}

	// Update global stats atomically
	constants.GlobalStats.Lock()
	constants.GlobalStats.LegacyCount = int64(p.Stats.LegacyCount)
	constants.GlobalStats.SegwitCount = int64(p.Stats.SegwitCount)
	constants.GlobalStats.NativeCount = int64(p.Stats.NativeCount)
	constants.GlobalStats.TotalCount = constants.GlobalStats.LegacyCount +
		constants.GlobalStats.SegwitCount +
		constants.GlobalStats.NativeCount
	constants.GlobalStats.Unlock()

	constants.AreAddressesLoaded = true
	return p, nil
}

// CountAddresses counts total addresses and updates stats without loading them all into memory
func (p *Parser) CountAddresses() (uint64, error) {
	iter := p.DB.NewIterator(nil, nil)
	defer iter.Release()

	count := uint64(0)
	// Reset stats before counting
	p.Stats.LegacyCount = 0
	p.Stats.SegwitCount = 0
	p.Stats.NativeCount = 0
	p.Stats.WScriptCount = 0

	batchSize := 0
	for iter.Next() {
		key := iter.Key()
		if len(key) > 5 && string(key[:5]) == "addr:" {
			addr := string(key[5:])
			addresses.CategorizeAddress(addr, p.Stats)
			count++
			batchSize++

			// Log progress every million addresses
			if batchSize >= 10_000_000 {
				logger.LogDebug(p.Logger, constants.LogDB,
					"Counted %s addresses...",
					utils.FormatWithCommas(int(count)))
				batchSize = 0
			}
		}
	}

	if err := iter.Error(); err != nil {
		return count, fmt.Errorf("error during address counting: %v", err)
	}

	return count, nil
}

func (p *Parser) CheckAddress(address string) (bool, int64) {
	p.addressMutex.RLock()
	hash := fastHash(address)
	balance, exists := p.addresses[hash]
	p.addressMutex.RUnlock()

	if !exists {
		return false, 0
	}

	return true, int64(balance)
}

// Address load verification
func (p *Parser) VerifyStats() {
	iter := p.DB.NewIterator(nil, nil)
	defer iter.Release()

	totalCount := uint64(0)
	for iter.Next() {
		key := iter.Key()
		if len(key) > 5 && string(key[:5]) == "addr:" {
			totalCount++
		}
	}

	if totalCount != p.AddressCount {
		logger.LogStatus(p.Logger, constants.LogWarn,
			"Address count mismatch: DB=%d, Stats=%d",
			totalCount, p.AddressCount)
		// Force recount
		p.LoadAllAddresses()
	}
}

// ----------------- OVERRIDES ------------------------------
//
// Add these methods to parser/parser.go

func (p *Parser) Cleanup() {
	if p.DB != nil {
		p.DB.Close()
	}
}

func (p *Parser) IsBlockProcessed(blockFile string) bool {
	p.processedBlocksMux.RLock()
	defer p.processedBlocksMux.RUnlock()
	return p.processedBlocks[blockFile]
}

func (p *Parser) MarkBlockProcessed(blockFile string) {
	p.processedBlocksMux.Lock()
	p.processedBlocks[blockFile] = true
	p.processedBlocksMux.Unlock()
}

func (p *Parser) LoadBlocksProcessed() error {
	iter := p.DB.NewIterator(nil, nil)
	defer iter.Release()

	prefix := []byte("block:")
	for iter.Next() {
		key := iter.Key()
		if bytes.HasPrefix(key, prefix) {
			blockFile := string(key[6:])
			p.processedBlocks[blockFile] = true
		}
	}
	return iter.Error()
}
