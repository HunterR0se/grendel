package runtime

import (
	"Grendel/constants"
	"Grendel/generator"
	"Grendel/logger"
	"Grendel/parser"
	"Grendel/utils"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"time"

	"github.com/btcsuite/btcd/btcec/v2"
	"github.com/shirou/gopsutil/mem"
)

// handleImport handles importing addresses from a compressed file.
// It returns an error if the import process fails.
func HandleImport(localLog *log.Logger) error {
	logger.LogHeaderStatus(localLog, constants.LogInfo, "Starting Address Import")

	// Initialize parser just for import
	importParser, err := parser.NewParser(localLog,
		filepath.Join(utils.GetBaseDir(), ".bitcoin"),
		filepath.Join(utils.GetBaseDir(), constants.AddressDBPath),
		true) // Force reparse during import

	if err != nil {
		return fmt.Errorf("failed to initialize parser for import: %w", err)
	}

	// Import addresses from the compressed file
	importPath := "./config/addresses.txt.gz"
	logger.LogStatus(localLog, constants.LogInfo, "Importing addresses from: %s", importPath)

	if err := importParser.ImportAddressesFromFile(importPath); err != nil {
		return fmt.Errorf("failed to import addresses: %w", err)
	}

	logger.LogStatus(localLog, constants.LogInfo, "Address import completed successfully")
	return nil
}

func verifyDatabase(p *parser.Parser) error {
	if p.DB == nil {
		return fmt.Errorf("database failed to initialize properly")
	}
	if _, err := p.DB.Has([]byte("test"), nil); err != nil {
		return fmt.Errorf("database access error: %w", err)
	}
	return nil
}

func logSystemInfo(ctx *AppContext) {
	v, _ := mem.VirtualMemory()
	logger.LogStatus(ctx.LocalLog, constants.LogVideo,
		"System has %d Cores and %.1f GB RAM",
		runtime.NumCPU(),
		float64(v.Total)/(1024*1024*1024))

	// Add GPU information logging
	if ctx.GpuInfo.Available {
		logger.LogStatus(ctx.LocalLog, constants.LogVideo,
			"%s %.0fGB VRAM (CUDA %s)",
			ctx.GpuInfo.Name,
			float64(ctx.GpuInfo.VRAM),
			utils.BoolToEnabledDisabled(ctx.GpuInfo.UsingCUDA))
	}

	logger.PrintSeparator(constants.LogVideo)
}

// checkMemoryUsage logs and manages memory usage.
func CheckMemoryUsage(ctx *AppContext) {
	v, _ := mem.VirtualMemory()
	usedPercentage := float64(v.Used) / float64(v.Total)

	if usedPercentage > constants.MemoryTarget {
		logger.LogStatus(ctx.LocalLog, constants.LogWarn,
			"Memory (%.1f%% > %.1f%% target) - %.1fGB/%.1fGB [Compacting]",
			usedPercentage*100, constants.MemoryTarget*100,
			float64(v.Used)/(1024*1024*1024),
			float64(v.Total)/(1024*1024*1024))
		go runtime.GC()
		debug.FreeOSMemory()
	}
}

// sendToChecker sends valid addresses to the address checker
// to compare generated addresses against addresses loaded from blocks
func SendToChecker(ctx *AppContext, address string, privateKey *btcec.PrivateKey, addrType generator.AddressType) {
	wallet := &WalletInfo{
		Address:    address,
		PrivateKey: privateKey,
		AddrType:   addrType,
	}

	// Try to send with backoff
	maxRetries := 3
	for i := 0; i < maxRetries; i++ {
		select {
		case ctx.AddressChan <- wallet:
			return
		default:
			time.Sleep(time.Millisecond * time.Duration(i*10))
		}
	}

	// Only increment drops after retries fail
	constants.TotalDroppedAddresses.Add(1)
}

// handleChannelFull logs and handles cases where the address channel is full.
func handleChannelFull(ctx *AppContext, address string) {
	logger.LogError(ctx.LocalLog, constants.LogError,
		fmt.Errorf("failed to send address '%s' to checker - channel full", address),
		"Address not sent due to full channel buffer")
}

func createBitcoinDir(baseDir string) error {
	return os.MkdirAll(filepath.Join(baseDir, ".bitcoin"), 0755)
}
