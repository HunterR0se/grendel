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

	"golang.org/x/exp/rand"
)

func Benchmark() {
	count := flag.Int("n", constants.GPUTestAddresses, "Number of keys to generate")
	flag.Parse()

	localLog := log.New(os.Stdout, "", 0)

	// Clear screen
	fmt.Print("\033[H\033[2J\n")
	logger.Banner()

	// Create GPU generator
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

	// Open the real address database
	dbPath := filepath.Join(utils.GetBaseDir(), constants.AddressDBPath)
	p, err := parser.NewParser(localLog, "", dbPath, false) // Note: false to not reparse
	if err != nil {
		logger.LogError(localLog, constants.LogError, err, "Failed to open address database")
		return
	}
	defer p.Cleanup()

	// Load addresses into memory
	logger.LogHeaderStatus(localLog,
		constants.LogDB,
		"Loading addresses into memory...")

	loadStart := time.Now()
	loadedCount, err := p.LoadAllAddresses()
	if err != nil {
		logger.LogError(localLog, constants.LogError, err, "Failed to load addresses")
		return
	}
	loadDuration := time.Since(loadStart)
	logger.LogStatus(localLog, constants.LogMem,
		"Loaded %s addresses in %.3f seconds",
		utils.FormatWithCommas(int(loadedCount)),
		loadDuration.Seconds())

	// MATCHING
	startTime := time.Now()
	matched := 0

	// Calculate progress intervals
	interval := *count / 10 // 10% intervals
	lastProgress := 0

	// Get all addresses into a slice for random selection
	allAddresses := p.GetAddresses()

	logger.LogStatus(localLog, constants.LogCheck,
		"Testing Matching against %s addresses",
		utils.FormatWithCommas(len(allAddresses)))
	logger.PrintSeparator(constants.LogCheck)

	for i := 0; i < *count; i++ {
		// Pick a random existing address
		randomIndex := rand.Intn(len(allAddresses))
		testAddr := allAddresses[randomIndex]

		if exists, _ := p.CheckAddress(testAddr); exists {
			matched++
		}

		// Show progress every 10%
		if i > 0 && i%interval == 0 && i/interval > lastProgress {
			progress := (i * 100) / *count
			currentRate := float64(i) / time.Since(startTime).Seconds()
			logger.LogStatus(localLog, constants.LogCheck,
				"Progress: %d%% (%s/sec)",
				progress,
				utils.FormatWithCommas(int(currentRate)))
			lastProgress = i / interval
		}
	}

	duration := time.Since(startTime)
	rate := float64(*count) / duration.Seconds()

	logger.LogHeaderStatus(localLog, constants.LogInfo,
		"Checked %s addresses in %.3f seconds",
		utils.FormatWithCommas(*count),
		duration.Seconds())
	logger.LogStatus(localLog, constants.LogInfo,
		"Matching rate: %s addresses/second",
		utils.FormatWithCommas(int(rate)))
	if matched > 0 {
		logger.LogStatus(localLog, constants.LogInfo,
			"Found %s matching addresses!",
			utils.FormatWithCommas(matched))
	}
	logger.PrintSeparator(constants.LogInfo)
}
