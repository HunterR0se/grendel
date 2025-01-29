package constants

import (
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/shirou/gopsutil/mem"
)

// Required for bitcoin new nodes - from debug.log
// var ObfuscationKey = []byte{0x7b, 0xc9, 0x9d, 0x89, 0xa8, 0x96, 0x4a, 0xdb}

// ObfuscationKey is the global variable holding the obfuscation key
var ObfuscationKey []byte

// Package-level variables
var (
	TrackAllAddresses     bool // track even zero balance btc addresses?
	Logger                *log.Logger
	DebugMode             bool
	GeneratorMode         bool
	LineLength            = 65 // max line length
	AreAddressesLoaded    bool
	TotalDroppedAddresses atomic.Int64
)

const (
	MinBufferSize = 512 * 1024 * 1024      // 128 * 1024 * 1024
	MaxBufferSize = 8 * 1024 * 1024 * 1024 // 1024 * 1024 * 1024
	MemoryTarget  = 0.95                   // Use x% of available memory
)

// Memory allocation proportions
const (
	BlockCacheProportion     = 0.70 // 60% of target memory
	WriteBufferProportion    = 0.20 // 25% of target memory
	CompactionSizeProportion = 0.10 // 15% of target memory
)

var (
	GPULocalBufferSize = 4096      // local size of GPU specific buffers
	GPUBatchBufferSize = 1_000_000 // 256 * 1024 // BATCH_SIZE 64M
	GPUTestAddresses   = 5_000_000 // total addresses for testing
)

// Specific to threads, memory management and processing
var (
	NumWorkers        = runtime.NumCPU() * 2 // 4x CPU
	RNGPoolSize       = 1024 * 1024          // 250_000
	ChannelBuffer     = 8 * 1024 * 1024      // 40_000
	MaxBlockFiles     = 1_000_000            // 1_000_000
	ImportBatchSize   = 500_000              // 500_000
	ImportLogInterval = 60 * time.Second     // Every 90 seconds
	// AddressCheckerBatchSize = 500_000              // new AddressCheckerBatchSize
)

var (
	AddressCheckerBatchSize = calculateOptimalBatchSize() // Dynamic sizing
)

// Initial map capacities based on typical Bitcoin node data
const (
	InitialProcessedBlocksCapacity = 2_000_000 // Increased from 750k
	InitialAddressesCapacity       = 5_000_000 // Doubled from 1M
	InitialCacheCapacity           = 500_000   // Doubled from 100k
)

// Config holds worker pool configuration
type Config struct {
	NumWorkers int
	BatchSize  int
	Logger     *log.Logger

	UseGPU bool
}

// File paths
const (
	AddressDBPath = ".bitcoin/addresses.db"
	BitcoinDir    = ".bitcoin"
)

// Categories
type AddressCategory struct {
	LegacyCount  int
	SegwitCount  int
	NativeCount  int
	WScriptCount int // Add this line
}

// Headers and text-based variables
var (
	LogStart  = "[⌛️ START] " // Good - startup
	LogStats  = "[📝 STATS] "  // Good - statistics
	LogParser = "[🧵 PARSE] "  // Good - parsing
	LogHeader = "[〰️ HEADR] " // header
	LogWarn   = "[⏰ ALARM] "  // Good - warning
	LogError  = "[❌ ERROR] "  // Good - error
	LogImport = "[📥  LOAD] "  // Good - importing
	LogDebug  = "[🔍 DEBUG] "  // Good - debugging
	LogLoaded = "[📦  DONE] "  // Should be ✅ for completion
	LogCheck  = "[✨ CHECK] "  // Good - checking
	LogRetry  = "[🔄 RETRY] "  // Good - retrying
	LogMem    = "[🧠 -MEM-] "  // Updated to use memory emoji
	LogInfo   = "[🔍  INFO] "  // Good - info
	LogDB     = "[📁 -DATA] "  // New prefix for database operations
	LogVideo  = "[🎮 -GPU-] "  // GPU acceleration

	// Status emojis for consistent usage
	EmojiFound   = "✨"  // Good - found items
	EmojiBalance = "💰"  // Good - balance
	EmojiKey     = "🔑"  // Good - keys
	EmojiSeed    = "🌱"  // Good - seeds
	EmojiDisk    = "💾"  // For disk operations
	EmojiMemory  = "🧠"  // For memory operations
	EmojiTime    = "⏱️" // Good - time
	EmojiAddress = "📫"  // Updated - addresses
	EmojiBitcoin = "₿"  // Good - bitcoin
	EmojiSuccess = "✅"  // Good - success
	EmojiError   = "❌"  // Good - error
	EmojiStats   = "🗃"  // Good - stats
	EmojiRocket  = "🚀"  // Good - startup
	EmojiImport  = "📥"  // Good - import
	EmojiWarning = "⚠️" // Good - warning
	EmojiBlocks  = "⛓️" // Updated - blockchain
	EmojiDB      = "📁"  // For database operations
	EmojiParser  = "⚡"  // For parser operations
	EmojiPath    = "🎯"  // For file paths
)

// GlobalStats
var (
	GlobalStats struct {
		sync.RWMutex
		// Address categories
		LegacyCount  int64
		SegwitCount  int64
		NativeCount  int64
		WScriptCount int64 // Add this line
		TotalCount   int64

		// Generator stats
		Generated uint64
		Found     uint64

		// Processing stats
		BlocksProcessed int64
		AddressesFound  int64

		// Metadata
		LastUpdated time.Time
	}
)

// Helper functions
func calculateOptimalBatchSize() int {
	v, _ := mem.VirtualMemory()
	if v == nil {
		return 500_000 // fallback to minimum if memory check fails
	}

	availableRAM := float64(v.Available)

	// Use 20% of available memory for batch operations
	// Each address takes approximately 43 bytes
	batchMemory := availableRAM * 0.20   // 20% of available memory in bytes
	optimalSize := int(batchMemory / 43) // Divide by bytes per address

	// Clamp between reasonable min/max values
	minBatch := 500_000    // Minimum 500k addresses per batch
	maxBatch := 20_000_000 // Maximum 20M addresses per batch

	if optimalSize < minBatch {
		optimalSize = minBatch
	}
	if optimalSize > maxBatch {
		optimalSize = maxBatch
	}

	// Log the calculated batch size in a clean format
	if Logger != nil {
		Logger.Printf("%s Batch Size: %d addresses",
			LogInfo,
			optimalSize)
	}

	return optimalSize
}

// Add helper functions to update stats
// Helper functions to update stats
func UpdateAddressStats(stats *AddressCategory) {
	GlobalStats.Lock()
	defer GlobalStats.Unlock()

	GlobalStats.LegacyCount = int64(stats.LegacyCount)
	GlobalStats.SegwitCount = int64(stats.SegwitCount)
	GlobalStats.NativeCount = int64(stats.NativeCount)
	GlobalStats.WScriptCount = int64(stats.WScriptCount) // Add this line
	GlobalStats.TotalCount = GlobalStats.LegacyCount +
		GlobalStats.SegwitCount +
		GlobalStats.NativeCount +
		GlobalStats.WScriptCount // Include WScriptCount in total
}

func IncrementGenerated() uint64 {
	GlobalStats.Lock()
	defer GlobalStats.Unlock()
	GlobalStats.Generated++
	return GlobalStats.Generated
}

func IncrementFound() uint64 {
	GlobalStats.Lock()
	defer GlobalStats.Unlock()
	GlobalStats.Found++
	return GlobalStats.Found
}
