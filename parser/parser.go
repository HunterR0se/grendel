package parser

import (
	"Grendel/addresses"
	"Grendel/constants"
	"Grendel/logger"
	"Grendel/utils"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
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
	addresses          map[string]string // Use a hash map for fast lookups
	addressMutex       sync.RWMutex
	BitcoinDir         string
	processedBlocksMux sync.RWMutex // for block parser
	Stats              *constants.AddressCategory
	processedBlocks    map[string]bool
	LastBlockHeight    int64
	BlocksProcessed    int64
	AddressesFound     int64
	lastUpdateTime     time.Time
}

func (p *Parser) CheckAddressBatch(addresses []string,
	results []bool,
	balances []int64) {

	p.addressMutex.RLock()
	// First pass: check in-memory cache
	for i, addr := range addresses {
		if value, exists := p.addresses[addr]; exists {
			results[i] = true
			balances[i] = int64(binary.LittleEndian.Uint64([]byte(value)))
		}
	}
	p.addressMutex.RUnlock()

	// Second pass: check database for missing addresses
	for i, addr := range addresses {
		if !results[i] {
			key := []byte("addr:" + addr)
			value, err := p.DB.Get(key, nil)
			if err == nil {
				results[i] = true
				balances[i] = int64(binary.LittleEndian.Uint64(value))

				// Add to memory cache
				p.addressMutex.Lock()
				p.addresses[addr] = string(value)
				p.addressMutex.Unlock()
			}
		}
	}
}

func calculateDBSize(dbPath string) float64 {
	files, err := os.ReadDir(dbPath)
	if err != nil {
		return 0
	}

	var totalSize int64
	for _, file := range files {
		info, err := file.Info()
		if err != nil {
			continue
		}
		// Count both .log and .ldb files
		if strings.HasSuffix(file.Name(), ".log") ||
			strings.HasSuffix(file.Name(), ".ldb") {
			totalSize += info.Size()
		}
	}
	return float64(totalSize) / (1024 * 1024 * 1024) // Convert to GB
}

func (p *Parser) AddTestAddress(addr string) error {
	p.addressMutex.Lock()
	defer p.addressMutex.Unlock()

	// Use a small test balance (0.0001 BTC in satoshis)
	testBalance := []byte{10, 0, 0, 0, 0, 0, 0, 0} // 10 satoshis

	p.addresses[addr] = string(testBalance)
	if err := p.DB.Put([]byte("addr:"+addr), testBalance, nil); err != nil {
		return err
	}

	addresses.CategorizeAddress(addr, p.Stats)
	p.AddressCount++
	return nil
}

// Lazy address loading (for speed) 20250123
func (p *Parser) LazyLoadAddresses() error {
	// Only load address statistics initially
	statsIter := p.DB.NewIterator(nil, nil)
	defer statsIter.Release()

	for statsIter.Next() {
		key := statsIter.Key()
		if len(key) > 5 && string(key[:5]) == "addr:" {
			addr := string(key[5:])
			addresses.CategorizeAddress(addr, p.Stats)
			p.AddressCount++
		}
	}

	// Don't load actual addresses into memory yet
	return nil
}

// ------------------------------------------------------------------
func NewParser(localLog *log.Logger,
	bitcoinDir,
	dbPath string,
	forceReparse bool) (*Parser, error) {

	// Get database size and estimate load time
	sizeGB := calculateDBSize(dbPath)
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

	// Use more conservative database options
	opts := &opt.Options{
		BlockCacheCapacity:     64 * 1024 * 1024, // Reduced to 64MB
		WriteBuffer:            32 * 1024 * 1024, // Reduced to 32MB
		CompactionTableSize:    16 * 1024 * 1024, // Reduced to 16MB
		OpenFilesCacheCapacity: 500,
		Filter:                 filter.NewBloomFilter(10),
		BlockRestartInterval:   8,
		BlockSize:              16 * 1024,
		WriteL0SlowdownTrigger: 8,
		WriteL0PauseTrigger:    24,
	}

	p := &Parser{
		Logger:          localLog,
		BitcoinDir:      bitcoinDir,
		DBPath:          dbPath,
		ForceReparse:    forceReparse,
		processedBlocks: make(map[string]bool, constants.InitialProcessedBlocksCapacity),
		// Start with smaller initial capacity
		addresses:       make(map[string]string, 100000),
		cache:           make(map[string]bool, constants.InitialCacheCapacity),
		Stats:           &constants.AddressCategory{},
		BlocksProcessed: 0,
		AddressesFound:  0,
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
			logger.LogError(localLog, constants.LogError, err, "Failed to reset processed blocks")
		}
	}

	// Instead of loading all addresses, just count them and update stats
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

	// Update global stats
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
	value, exists := p.addresses[address]
	p.addressMutex.RUnlock()

	if !exists {
		return false, 0
	}

	return true, int64(binary.LittleEndian.Uint64([]byte(value)))
}

func (p *Parser) working_CheckAddress(address string) (bool, int64) {
	start := time.Now()

	p.addressMutex.RLock()
	// Check memory first
	if value, exists := p.addresses[address]; exists {
		p.addressMutex.RUnlock()
		memTime := time.Since(start)
		if constants.DebugMode {
			fmt.Printf("Memory lookup took: %v\n", memTime)
		}
		return true, int64(binary.LittleEndian.Uint64([]byte(value)))
	}
	p.addressMutex.RUnlock()

	// If not in memory, check database
	dbStart := time.Now()
	key := []byte("addr:" + address)
	value, err := p.DB.Get(key, nil)
	dbTime := time.Since(dbStart)
	if constants.DebugMode {
		fmt.Printf("DB lookup took: %v\n", dbTime)
	}

	if err != nil {
		return false, 0
	}

	// Add to memory cache
	cacheStart := time.Now()
	p.addressMutex.Lock()
	p.addresses[address] = string(value)
	p.addressMutex.Unlock()
	cacheTime := time.Since(cacheStart)
	if constants.DebugMode {
		fmt.Printf("Cache update took: %v\n", cacheTime)
	}

	totalTime := time.Since(start)
	if constants.DebugMode {
		fmt.Printf("Total check took: %v\n", totalTime)
	}

	return true, int64(binary.LittleEndian.Uint64(value))
}

// Modified CheckAddress to load addresses on demand
func (p *Parser) _CheckAddress(address string) (bool, int64) {
	p.addressMutex.RLock()

	// Check memory first
	if value, exists := p.addresses[address]; exists {
		p.addressMutex.RUnlock()
		return true, int64(binary.LittleEndian.Uint64([]byte(value)))
	}
	p.addressMutex.RUnlock()

	// If not in memory, check database
	key := []byte("addr:" + address)
	value, err := p.DB.Get(key, nil)
	if err != nil {
		return false, 0
	}

	// Add to memory cache
	p.addressMutex.Lock()
	p.addresses[address] = string(value)
	p.addressMutex.Unlock()

	return true, int64(binary.LittleEndian.Uint64(value))
}

// ---------------------------------------------

func (p *Parser) LoadBlocksProcessed() error {
	key := []byte("blocks_processed")
	value, err := p.DB.Get(key, nil)
	if err == leveldb.ErrNotFound {
		p.BlocksProcessed = 0
		return nil
	}
	if err != nil {
		return err
	}
	p.BlocksProcessed = int64(binary.LittleEndian.Uint64(value))
	return nil
}

func (p *Parser) SaveBlocksProcessed() error {
	key := []byte("blocks_processed")
	value := make([]byte, 8)
	binary.LittleEndian.PutUint64(value, uint64(p.BlocksProcessed))
	return p.DB.Put(key, value, nil)
}

func (p *Parser) IsBlockProcessed(blockFile string) bool {
	// Always check ForceReparse first
	if p.ForceReparse {
		if constants.DebugMode {
			p.Logger.Printf("[🔍 DEBUG] Force reparse enabled, treating block %s as unprocessed",
				blockFile)
		}
		return false
	}

	p.processedBlocksMux.RLock()
	defer p.processedBlocksMux.RUnlock()

	// Check memory cache first
	if processed, exists := p.processedBlocks[blockFile]; exists {
		return processed && !p.ForceReparse // Add ForceReparse check here too
	}

	// Then check database
	key := append([]byte("block:"), blockFile...)
	exists, err := p.DB.Has(key, nil)
	if err != nil {
		p.Logger.Printf("[❌ ERROR] Error checking block status: %v", err)
		return false
	}

	// Cache the result, but respect ForceReparse
	if exists && !p.ForceReparse {
		p.processedBlocks[blockFile] = true
	} else {
		p.processedBlocks[blockFile] = false
	}

	return exists && !p.ForceReparse
}

func (p *Parser) MarkBlockProcessed(blockFile string) error {
	p.processedBlocksMux.Lock()
	defer p.processedBlocksMux.Unlock()

	if constants.DebugMode {
		p.Logger.Printf("%s Marking block %s as processed",
			constants.LogDebug, blockFile)
	}

	// Mark in memory
	p.processedBlocks[blockFile] = true

	// Mark in database
	key := append([]byte("block:"), blockFile...)
	if err := p.DB.Put(key, []byte{1}, nil); err != nil {
		return fmt.Errorf("failed to mark block as processed: %v", err)
	}

	return nil
}

func (p *Parser) saveProgress() error {
	// Save current block height
	heightBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(heightBytes, uint64(p.BlockHeight))
	if err := p.DB.Put([]byte("height"), heightBytes, nil); err != nil {
		return fmt.Errorf("failed to save height: %v", err)
	}
	return nil
}

// Add cleanup method for interrupted scans
func (p *Parser) Cleanup() error {
	if p == nil {
		return nil
	}

	// Save current progress
	if err := p.saveProgress(); err != nil {
		p.Logger.Printf("%s Failed to save progress during cleanup: %v", constants.LogError, err)
	}

	// Force garbage collection
	runtime.GC()
	debug.FreeOSMemory()

	return nil
}

func (p *Parser) ResetProcessedBlocks() error {
	if !p.ForceReparse {
		return nil
	}

	p.processedBlocksMux.Lock()
	defer p.processedBlocksMux.Unlock()

	// Clear memory cache
	p.processedBlocks = make(map[string]bool, constants.InitialProcessedBlocksCapacity)

	// Clear database entries
	iter := p.DB.NewIterator(nil, nil)
	defer iter.Release()

	batch := new(leveldb.Batch)
	for iter.Next() {
		key := iter.Key()
		if len(key) > 6 && string(key[:6]) == "block:" {
			batch.Delete(key)
		}
	}

	if err := p.DB.Write(batch, nil); err != nil {
		return fmt.Errorf("failed to clear processed blocks: %v", err)
	}

	p.BlocksProcessed = 0
	return p.SaveBlocksProcessed()
}

func (p *Parser) loadStats() error {
	// Load stats from database
	statsKey := []byte("address_stats")
	data, err := p.DB.Get(statsKey, nil)
	if err == leveldb.ErrNotFound {
		return nil // No stats saved yet
	}
	if err != nil {
		return err
	}

	// Update global stats after loading
	constants.UpdateAddressStats(p.Stats)
	return json.Unmarshal(data, p.Stats)
}

func (p *Parser) LoadAllAddresses() (uint64, error) {
	iter := p.DB.NewIterator(nil, nil)
	defer iter.Release()

	// Pre-allocate map with a good size estimate
	p.addresses = make(map[string]string, p.AddressCount)

	// Prepare the prefix once
	prefix := []byte("addr:")
	count := uint64(0)

	// Local counters to avoid locks
	var legacy, segwit, native int

	// Batch processing
	for iter.Next() {
		key := iter.Key()
		if len(key) > 5 && bytes.Equal(key[:5], prefix) {
			// Direct slice to string conversion for key
			addr := string(key[5:])

			// Quick categorization without regex
			switch {
			case strings.HasPrefix(addr, "1"):
				legacy++
			case strings.HasPrefix(addr, "3"):
				segwit++
			case strings.HasPrefix(addr, "bc1"):
				native++
			}

			// Store value directly without conversion
			p.addresses[addr] = string(iter.Value())
			count++
		}
	}

	// Update stats once at the end
	p.Stats.LegacyCount = legacy
	p.Stats.SegwitCount = segwit
	p.Stats.NativeCount = native

	// Single lock for global stats update
	constants.GlobalStats.Lock()
	constants.GlobalStats.LegacyCount = int64(legacy)
	constants.GlobalStats.SegwitCount = int64(segwit)
	constants.GlobalStats.NativeCount = int64(native)
	constants.GlobalStats.TotalCount = constants.GlobalStats.LegacyCount +
		constants.GlobalStats.SegwitCount +
		constants.GlobalStats.NativeCount
	constants.GlobalStats.Unlock()

	p.AddressCount = count
	return count, nil
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

// Optimizations
//
// GetAddresses returns a slice of all stored addresses
func (p *Parser) GetAddresses() []string {
	addresses := make([]string, 0, len(p.addresses))
	for addr := range p.addresses {
		addresses = append(addresses, addr)
	}
	return addresses
}
