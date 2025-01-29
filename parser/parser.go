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

// Optimized CheckAddressBatch
func (p *Parser) CheckAddressBatch(addresses []string, results []bool, balances []int64) {
	// Pre-check array sizes
	if len(addresses) != len(results) || len(addresses) != len(balances) {
		return
	}

	// Single read lock for entire batch
	p.addressMutex.RLock()
	defer p.addressMutex.RUnlock()

	// Process in cache-friendly chunks
	const chunkSize = 256 // Tuned for typical L1 cache size
	for i := 0; i < len(addresses); i += chunkSize {
		end := i + chunkSize
		if end > len(addresses) {
			end = len(addresses)
		}

		// Pre-calculate hashes for the chunk
		hashes := make([]uint64, end-i)
		for j := range hashes {
			hashes[j] = fastHash(addresses[i+j])
		}

		// Check addresses using pre-calculated hashes
		for j := range hashes {
			if balance, exists := p.addresses[hashes[j]]; exists {
				results[i+j] = true
				balances[i+j] = int64(balance)
			}
		}
	}
}

// Optimized fastHash - uses xxHash algorithm for better performance
func fastHash(s string) uint64 {
	// xxHash constants
	const (
		prime1 = 11400714785074694791
		prime2 = 14029467366897019727
		prime3 = 1609587929392839161
		prime4 = 9650029242287828579
		prime5 = 2870177450012600261
	)

	var h64 uint64 = prime5
	data := []byte(s)
	i := 0

	// Process 8 bytes at a time
	for ; i+8 <= len(data); i += 8 {
		k1 := uint64(data[i]) | uint64(data[i+1])<<8 |
			uint64(data[i+2])<<16 | uint64(data[i+3])<<24 |
			uint64(data[i+4])<<32 | uint64(data[i+5])<<40 |
			uint64(data[i+6])<<48 | uint64(data[i+7])<<56

		k1 *= prime2
		k1 = (k1 << 31) | (k1 >> 33)
		k1 *= prime1

		h64 ^= k1
		h64 = (h64 << 27) | (h64 >> 37)
		h64 = h64*prime1 + prime4
	}

	// Process remaining bytes
	for ; i < len(data); i++ {
		h64 ^= uint64(data[i])
		h64 = (h64 << 23) | (h64 >> 41)
		h64 *= prime2
		h64 ^= prime3
	}

	// Final mix
	h64 ^= uint64(len(data))
	h64 = (h64 ^ (h64 >> 33)) * prime2
	h64 = (h64 ^ (h64 >> 29)) * prime3
	h64 = h64 ^ (h64 >> 32)

	return h64
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

// Optimized LoadAllAddresses with better memory management and performance
func (p *Parser) LoadAllAddresses() (uint64, error) {
	startTime := time.Now()

	// Pre-allocate map with better capacity planning
	const mapLoadFactor = 0.75
	initialCapacity := uint64(float64(p.AddressCount) / mapLoadFactor)

	p.addressMutex.Lock()
	p.addresses = make(map[uint64]uint64, initialCapacity)
	p.addressMutex.Unlock()

	// Use larger batch size for better throughput
	var batchSize = constants.ImportBatchSize
	batch := make([]struct {
		hash    uint64
		balance uint64
	}, 0, batchSize)

	iter := p.DB.NewIterator(nil, nil)
	defer iter.Release()

	prefix := []byte("addr:")
	count := uint64(0)
	lastLogTime := time.Now()
	var lastCount uint64

	// Process in batches for better memory efficiency
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

		// Bulk insert when batch is full
		if len(batch) >= batchSize {
			p.addressMutex.Lock()
			for _, item := range batch {
				p.addresses[item.hash] = item.balance
				count++
			}
			p.addressMutex.Unlock()
			batch = batch[:0]

			// Log progress with rate calculation every 5 seconds
			if time.Since(lastLogTime) >= 5*time.Second {
				rate := float64(count-lastCount) / time.Since(lastLogTime).Seconds()
				progress := float64(count) / float64(p.AddressCount) * 100

				logger.LogStatus(p.Logger, constants.LogDB,
					"%11s addresses (%4.1f%%) at %8.0f/sec",
					utils.FormatWithCommas(int(count)),
					progress,
					rate)

				lastCount = count
				lastLogTime = time.Now()
			}
		}
	}

	// Process remaining addresses
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

	elapsed := time.Since(startTime)
	rate := float64(count) / elapsed.Seconds()

	logger.LogDebug(p.Logger, constants.LogLoaded,
		"Loaded %s addresses in %.1f seconds (%.0f/sec)",
		utils.FormatWithCommas(int(count)),
		elapsed.Seconds(),
		rate)

	return count, nil
}

// Optimized VerifyStats
func (p *Parser) VerifyStats() {
	iter := p.DB.NewIterator(nil, nil)
	defer iter.Release()

	totalCount := uint64(0)
	prefix := []byte("addr:")
	startTime := time.Now()
	lastLogTime := time.Now()
	var lastCount uint64

	// Keep track of address types for verification
	stats := &constants.AddressCategory{}

	for iter.Next() {
		key := iter.Key()
		if len(key) > 5 && bytes.Equal(key[:5], prefix) {
			addr := string(key[5:])
			addresses.CategorizeAddress(addr, stats)
			totalCount++

			// Log progress with rate calculation
			if totalCount%uint64(constants.AddressCheckerBatchSize) == 0 {
				now := time.Now()
				if now.Sub(lastLogTime) >= 5*time.Second {
					rate := float64(totalCount-lastCount) / now.Sub(lastLogTime).Seconds()
					memStats := utils.GetMemStats()

					logger.LogStatus(p.Logger, constants.LogDB,
						"Verified %11s (%8.0f/sec) %.1f GB",
						utils.FormatWithCommas(int(totalCount)),
						rate,
						memStats.AllocatedGB)

					lastCount = totalCount
					lastLogTime = now
				}
			}
		}
	}

	if err := iter.Error(); err != nil {
		logger.LogError(p.Logger, constants.LogError, err,
			"Error during stats verification")
		return
	}

	// Check for significant discrepancy
	if totalCount != p.AddressCount {
		discrepancy := math.Abs(float64(totalCount) - float64(p.AddressCount))
		percentDiff := (discrepancy / float64(p.AddressCount)) * 100

		logger.LogStatus(p.Logger, constants.LogWarn,
			"Address count mismatch: DB=%s, Stats=%s (%.1f%% difference)",
			utils.FormatWithCommas(int(totalCount)),
			utils.FormatWithCommas(int(p.AddressCount)),
			percentDiff)

		// Force recount if difference is significant (>1%)
		if percentDiff > 1.0 {
			logger.LogStatus(p.Logger, constants.LogWarn,
				"Significant count mismatch detected, triggering reload")
			p.LoadAllAddresses()
		}
	}

	// Update global stats atomically
	constants.GlobalStats.Lock()
	constants.GlobalStats.LegacyCount = int64(stats.LegacyCount)
	constants.GlobalStats.SegwitCount = int64(stats.SegwitCount)
	constants.GlobalStats.NativeCount = int64(stats.NativeCount)
	constants.GlobalStats.WScriptCount = int64(stats.WScriptCount)
	constants.GlobalStats.TotalCount = constants.GlobalStats.LegacyCount +
		constants.GlobalStats.SegwitCount +
		constants.GlobalStats.NativeCount +
		constants.GlobalStats.WScriptCount
	constants.GlobalStats.LastUpdated = time.Now()
	constants.GlobalStats.Unlock()

	elapsed := time.Since(startTime)
	rate := float64(totalCount) / elapsed.Seconds()

	logger.LogHeaderStatus(p.Logger, constants.LogDB,
		"* %s addresses %.1f secs (%.0f/sec)",
		utils.FormatWithCommas(int(totalCount)),
		elapsed.Seconds(),
		rate)
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
