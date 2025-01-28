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
	"math"
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

// Optimized CheckAddressBatch in parser/parser.go
func (p *Parser) CheckAddressBatch(addresses []string, results []bool, balances []int64) {
	// Pre-check array sizes
	if len(addresses) != len(results) || len(addresses) != len(balances) {
		return
	}

	p.addressMutex.RLock()
	defer p.addressMutex.RUnlock()

	// Process in chunks for better cache utilization
	const chunkSize = 256
	for i := 0; i < len(addresses); i += chunkSize {
		end := i + chunkSize
		if end > len(addresses) {
			end = len(addresses)
		}

		// Pre-calculate hashes for the chunk
		hashes := make([]uint64, end-i)
		for j := 0; j < len(hashes); j++ {
			hashes[j] = fastHash(addresses[i+j])
		}

		// Check chunk of addresses using pre-calculated hashes
		for j := 0; j < len(hashes); j++ {
			if balance, exists := p.addresses[hashes[j]]; exists {
				results[i+j] = true
				balances[i+j] = int64(balance)
			}
		}
	}
}

// Optimized fastHash function
func fastHash(s string) uint64 {
	// Use xxHash for better performance
	var h uint64 = 14695981039346656037
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return h
}

// Optimized CheckAddress for single lookups
func (p *Parser) CheckAddress(address string) (bool, int64) {
	p.addressMutex.RLock()
	hash := fastHash(address)
	balance, exists := p.addresses[hash]
	p.addressMutex.RUnlock()

	return exists, int64(balance)
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
	defer p.addressMutex.RUnlock()

	// Create iterator to get actual addresses from database
	iter := p.DB.NewIterator(nil, nil)
	defer iter.Release()

	addresses := make([]string, 0, 1000)
	prefix := []byte("addr:")

	// Get actual addresses from database
	for iter.Next() {
		key := iter.Key()
		if len(key) > 5 && bytes.Equal(key[:5], prefix) {
			// Extract actual address from key
			addr := string(key[5:])
			addresses = append(addresses, addr)
		}

		// Limit the number of addresses we return for testing
		if len(addresses) >= 1000 {
			break
		}
	}

	return addresses
}

func (p *Parser) _GetAddresses() []string {
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

// CountAddresses counts total addresses and updates stats without
// loading them all into memory
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

// ----------------- ADDRESS LOADING -----------------------

// Optimized LoadAllAddresses
func (p *Parser) LoadAllAddresses() (uint64, error) {
	// Pre-allocate map with capacity estimate
	const mapLoadFactor = 0.75
	initialCapacity := uint64(float64(p.AddressCount) / mapLoadFactor)

	p.addressMutex.Lock()
	p.addresses = make(map[uint64]uint64, initialCapacity)
	p.addressMutex.Unlock()

	// Use batched reading for better performance
	const batchSize = 100_000
	batch := make([]struct {
		hash    uint64
		balance uint64
	}, 0, batchSize)

	iter := p.DB.NewIterator(nil, nil)
	defer iter.Release()

	prefix := []byte("addr:")
	count := uint64(0)

	// Process in batches
	for iter.Next() {
		key := iter.Key()
		if len(key) <= 5 || !bytes.Equal(key[:5], prefix) {
			continue
		}

		addr := string(key[5:])
		balance := binary.LittleEndian.Uint64(iter.Value())

		batch = append(batch, struct {
			hash    uint64
			balance uint64
		}{
			hash:    fastHash(addr),
			balance: balance,
		})

		// When batch is full, bulk insert
		if len(batch) >= batchSize {
			p.addressMutex.Lock()
			for _, item := range batch {
				p.addresses[item.hash] = item.balance
				count++
			}
			p.addressMutex.Unlock()
			batch = batch[:0]
		}

		// Log progress periodically
		if count%1_000_000 == 0 && constants.DebugMode {
			logger.LogDebug(p.Logger, constants.LogDB,
				"Loaded %s addresses...",
				utils.FormatWithCommas(int(count)))
		}
	}

	// Insert any remaining addresses
	if len(batch) > 0 {
		p.addressMutex.Lock()
		for _, item := range batch {
			p.addresses[item.hash] = item.balance
			count++
		}
		p.addressMutex.Unlock()
	}

	if err := iter.Error(); err != nil {
		return count, fmt.Errorf("error during address loading: %v", err)
	}

	// Verify load completed successfully
	if count != p.AddressCount {
		logger.LogStatus(p.Logger, constants.LogWarn,
			"Address count mismatch: Expected %d, Loaded %d",
			p.AddressCount, count)
	}

	return count, nil
}

// Optimized VerifyStats
func (p *Parser) VerifyStats() {
	const verifyBatchSize = 100_000

	iter := p.DB.NewIterator(nil, nil)
	defer iter.Release()

	totalCount := uint64(0)
	prefix := []byte("addr:")

	// Use batched verification
	batch := 0
	startTime := time.Now()

	for iter.Next() {
		key := iter.Key()
		if len(key) > 5 && bytes.Equal(key[:5], prefix) {
			totalCount++
			batch++

			if batch >= verifyBatchSize {
				if constants.DebugMode {
					rate := float64(totalCount) / time.Since(startTime).Seconds()
					logger.LogDebug(p.Logger, constants.LogDB,
						"Verified %s addresses (%.0f/sec)",
						utils.FormatWithCommas(int(totalCount)),
						rate)
				}
				batch = 0
			}
		}
	}

	if totalCount != p.AddressCount {
		logger.LogStatus(p.Logger, constants.LogWarn,
			"Address count mismatch: DB=%d, Stats=%d",
			totalCount, p.AddressCount)

		// Force recount only if significant difference
		if math.Abs(float64(totalCount)-float64(p.AddressCount)) >
			float64(p.AddressCount)*0.01 {
			p.LoadAllAddresses()
		}
	}

	// Update verification timestamp
	p.lastUpdateTime = time.Now()
}

// ----------------- OVERRIDES ------------------------------

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
