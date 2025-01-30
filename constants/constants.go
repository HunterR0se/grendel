package constants

import (
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/shirou/gopsutil/mem"
)

func init() {
	// Sanity check buffer sizes
	if ChannelBuffer > 10_000_000 {
		log.Printf("Warning: Large channel buffer size: %d", ChannelBuffer)
	}
	if AddressCheckerBatchSize > 25_000_000 {
		log.Printf("Warning: Large batch size: %d", AddressCheckerBatchSize)
	}
}

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
	MinBufferSize = 256 * 1024 * 1024      // Reduced from 512MB to 256MB
	MaxBufferSize = 4 * 1024 * 1024 * 1024 // Reduced from 8GB to 4GB
	MemoryTarget  = 0.70                   // Reduced from 0.95 to 0.70 (70% of available memory)
)

// Memory allocation proportions - adjusted for better balance
const (
	BlockCacheProportion     = 0.50 // Reduced from 0.70 to 0.50 (50% of target memory)
	WriteBufferProportion    = 0.15 // Reduced from 0.20 to 0.15 (15% of target memory)
	CompactionSizeProportion = 0.05 // Reduced from 0.10 to 0.05 (5% of target memory)
)

// Reduce batch and buffer sizes
var (
	GPULocalBufferSize = 4096      // Keep as is
	GPUBatchBufferSize = 500_000   // Reduced from 1M to 500k
	GPUTestAddresses   = 1_000_000 // Reduced from 5M to 1M
)

// Specific to threads, memory management and processing
var (
	NumWorkers        = runtime.NumCPU() // Reduced from 2x to 1x CPU
	RNGPoolSize       = 256 * 1024       // Reduced from 1M to 256k
	ChannelBuffer     = 2 * 1024 * 1024  // Reduced from 8M to 2M
	MaxBlockFiles     = 500_000          // Reduced from 1M to 500k
	ImportBatchSize   = 250_000          // Reduced from 500k to 250k
	ImportLogInterval = 60 * time.Second // Keep as is
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

// Modify calculateOptimalBatchSize() to be more conservative
func calculateOptimalBatchSize() int {
	v, _ := mem.VirtualMemory()
	if v == nil {
		return 250_000 // reduced fallback value
	}

	availableRAM := float64(v.Available)

	// Use 15% of available memory for batch operations (reduced from 20%)
	// Each address takes approximately 43 bytes
	batchMemory := availableRAM * 0.15
	optimalSize := int(batchMemory / 43)

	// More conservative min/max values
	minBatch := 250_000   // Reduced from 500k
	maxBatch := 5_000_000 // Reduced from 20M to 5M

	if optimalSize < minBatch {
		optimalSize = minBatch
	}
	if optimalSize > maxBatch {
		optimalSize = maxBatch
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
