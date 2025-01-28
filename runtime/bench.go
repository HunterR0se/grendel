package runtime

import (
	"Grendel/constants"
	"Grendel/generator"
	"Grendel/gpu"
	"Grendel/logger"
	"Grendel/parser"
	"Grendel/utils"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

func Benchmark() {
	// Single flag for test count
	count := flag.Int("n", constants.GPUTestAddresses, "Number of keys to generate")
	flag.Parse()

	// Initialize logging once
	localLog := log.New(os.Stdout, "", 0)

	// Clear screen and show banner
	fmt.Print("\033[H\033[2J\n")
	logger.Banner()

	// Initialize GPU with error handling
	gen, err := gpu.NewCUDAGenerator(&generator.Config{})
	if err != nil {
		logger.LogError(localLog, constants.LogError, err, "Failed to create GPU generator")
		return
	}
	defer gen.Close()

	// Run generation benchmark
	if err := gpu.RunBenchmark(*count, gen); err != nil {
		logger.LogError(localLog, constants.LogError, err, "Benchmark failed")
		return
	}

	// Open address database with minimal config
	dbPath := filepath.Join(utils.GetBaseDir(), constants.AddressDBPath)
	p, err := parser.NewParser(localLog, "", dbPath, false) // false = don't reparse
	if err != nil {
		logger.LogError(localLog, constants.LogError, err, "Failed to open address database")
		return
	}
	defer p.Cleanup()

	// Load addresses with performance monitoring
	logger.LogHeaderStatus(localLog, constants.LogDB, "Loading addresses into memory...")

	// Time address loading
	loadStart := time.Now()
	loadedCount, err := p.LoadAllAddresses()
	if err != nil {
		logger.LogError(localLog, constants.LogError, err, "Failed to load addresses")
		return
	}

	// Log load performance
	logger.LogStatus(localLog, constants.LogMem,
		"Loaded %s addresses in %.3f seconds",
		utils.FormatWithCommas(int(loadedCount)),
		time.Since(loadStart).Seconds())

	// Create minimal context
	ctx := &AppContext{
		LocalLog: localLog,
		Parser:   p,
	}

	// Run address checking
	matched, duration, rate, err := CheckAddresses(ctx, *count)
	if err != nil {
		logger.LogError(localLog, constants.LogError, err, "Address checking failed")
		return
	}

	// Report results
	ReportAddressCheckResults(ctx, *count, matched, duration, rate)
}
