package main

import (
	"Grendel/constants"
	"Grendel/gpu"
	"Grendel/loader"
	"Grendel/logger"
	"Grendel/parser"
	"Grendel/utils"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"golang.org/x/exp/rand"

	"github.com/shirou/gopsutil/mem"
)

func initialize(genMode, debugMode, forceReparse bool, gpuInfo gpu.GPUInfo, cpuMode, gpuMode bool) (*AppContext, gpu.Generator, error) {
	ctx := &AppContext{
		localLog:     log.New(os.Stdout, "", 0),
		shutdownChan: make(chan struct{}),
		genMode:      genMode,
		debugMode:    debugMode,
		forceReparse: forceReparse,
		doneChan:     make(chan struct{}), // shutdown
		wg:           sync.WaitGroup{},    // waitgroup
		gpuInfo:      gpuInfo,             // for GPU
		cpuMode:      cpuMode,
		gpuMode:      gpuMode, // Add this line
	}

	// Set up logger constant
	constants.Logger = log.New(os.Stdout, "", log.LstdFlags)
	constants.GeneratorMode = ctx.genMode

	if ctx.debugMode {
		constants.DebugMode = true
	}

	// Check GPU availability
	logSystemInfo(ctx)

	// Setup paths
	ctx.baseDir = getBaseDir()
	ctx.parserPath = filepath.Join(ctx.baseDir, ".bitcoin")
	ctx.dbPath = filepath.Join(ctx.baseDir, constants.AddressDBPath)
	ctx.addressPath = filepath.Join(ctx.baseDir, constants.KnownAddressesPath)

	// Create bitcoin directory if needed
	if err := createBitcoinDir(ctx.baseDir); err != nil {
		return nil, nil, fmt.Errorf("failed to create bitcoin directory: %w", err)
	}

	// Initialize parser
	newParser, err := parser.NewParser(ctx.localLog, ctx.parserPath, ctx.dbPath, ctx.forceReparse)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to initialize parser: %w", err)
	}
	ctx.parser = newParser

	// Initialize parser stats
	ctx.parser.Stats = &constants.AddressCategory{}

	// Verify database
	if err := verifyDatabase(ctx.parser); err != nil {
		return nil, nil, fmt.Errorf("database verification failed: %w", err)
	}

	// Set up shutdown handler
	ctx.shutdownChan = setupGracefulShutdown(ctx.localLog, ctx.shutdownChan, ctx.doneChan)

	// Initialize generator
	var generatorInstance gpu.Generator
	if ctx.cpuMode {
		generatorInstance, err = gpu.NewCPUGenerator(nil)
	} else if ctx.gpuMode && ctx.gpuInfo.Available && ctx.gpuInfo.UsingCUDA {
		generatorInstance, err = gpu.NewCUDAGenerator(nil)
	} else {
		generatorInstance, err = gpu.NewCPUGenerator(nil) // Default to CPU
	}
	if err != nil {
		return nil, nil, fmt.Errorf("failed to initialize generator: %v", err)
	}

	ctx.generator = generatorInstance

	return ctx, generatorInstance, nil
}

// 2. Load blocks
func loadBlocks(ctx *AppContext) error {
	logger.LogHeaderStatus(ctx.localLog, constants.LogDB, "Starting the Block Loader...")
	logger.PrintSeparator(constants.LogDB)

	ctx.blockLoader = loader.NewBlockLoader(ctx.parser)

	blockFiles, err := ctx.blockLoader.FindBlockFiles()
	if err != nil {
		return fmt.Errorf("failed to scan for blocks: %w", err)
	}

	if len(blockFiles) == 0 {
		logger.LogStatus(ctx.localLog, constants.LogDB, "No new blocks found to process")
		return startBackgroundLoader(ctx)
	}

	logger.LogDebug(ctx.localLog, constants.LogDB,
		"Found %d blocks to process", len(blockFiles))

	if err := ctx.blockLoader.LoadNewBlocks(); err != nil {
		logger.LogError(ctx.localLog, constants.LogError, err,
			"Block loading encountered errors but continuing...")
	}

	if !constants.AreAddressesLoaded {
		count, err := ctx.parser.LoadAllAddresses()
		if err != nil {
			logger.LogError(ctx.localLog, constants.LogError, err,
				"Failed to load addresses from database")
		} else {
			logger.LogStatus(ctx.localLog, constants.LogDB,
				"Loaded %s addresses from database", utils.FormatWithCommas(int(count)))
			ctx.addressesLoaded = true
		}

		ctx.blockLoader.LogAddressSummary("Block Processing Complete", true)
	}

	logger.LogStatus(ctx.localLog, constants.LogDB, "Block loader successfully started.")
	return startBackgroundLoader(ctx)
}

func startBackgroundLoader(ctx *AppContext) error {
	// Create a map to track processed blocks using their base names
	processedBlocks := make(map[string]bool)

	// Record initially processed blocks by their base names
	initialFiles, err := ctx.blockLoader.FindBlockFiles()
	if err == nil {
		for _, file := range initialFiles {
			baseName := filepath.Base(file)
			if ctx.parser.IsBlockProcessed(file) {
				processedBlocks[baseName] = true
				if constants.DebugMode {
					logger.LogDebug(ctx.localLog, constants.LogDB,
						"Initially marking as processed: %s", baseName)
				}
			}
		}
	}

	go func() {
		ticker := time.NewTicker(30 * time.Minute)
		defer ticker.Stop()

		lastScan := time.Now()

		for {
			select {
			case <-ctx.shutdownChan:
				return
			case <-ticker.C:
				// Only scan if enough time has passed
				if time.Since(lastScan) < 4*time.Minute {
					continue
				}

				// Find any new blocks
				currentFiles, err := ctx.blockLoader.FindBlockFiles()
				if err != nil {
					logger.LogError(ctx.localLog, constants.LogError, err,
						"Failed to scan for new blocks")
					continue
				}

				// Only process blocks we haven't seen before AND aren't processed
				var newBlocks []string
				for _, file := range currentFiles {
					baseName := filepath.Base(file)
					if !processedBlocks[baseName] {
						if !ctx.parser.IsBlockProcessed(file) {
							newBlocks = append(newBlocks, file)
							processedBlocks[baseName] = true
							if constants.DebugMode {
								logger.LogDebug(ctx.localLog, constants.LogDB,
									"Found new block to process: %s", baseName)
							}
						}
					}
				}

				if len(newBlocks) > 0 {
					logger.LogDebug(ctx.localLog, constants.LogDB,
						"Processing %d new blocks", len(newBlocks))

					// Process only the new blocks directly
					err = ctx.blockLoader.ProcessNewBlocks(
						newBlocks,
						ctx.parser.BlocksProcessed,
						ctx.parser.AddressesFound)

					if err != nil {
						logger.LogError(ctx.localLog, constants.LogError, err,
							"Failed to process new blocks")
					}
				}

				lastScan = time.Now()
			}
		}
	}()

	return nil
}

// 3. Start address generation
func startAddressGeneration(ctx *AppContext) error {
	if ctx.generator == nil {
		return fmt.Errorf("generator not initialized")
	}

	ctx.memLimits = utils.CalculateMemoryLimits()

	availDisk, totalDisk := utils.GetAvailableDiskSpace()
	logger.LogHeaderStatus(ctx.localLog, constants.LogMem,
		"CPU Cores: %4d Disk: %7.1fGB Free:  %6.1fGB",
		runtime.NumCPU(),
		totalDisk, availDisk)

	logger.LogStatus(ctx.localLog, constants.LogMem,
		"Total: %6.1fGB Avail: %6.1fGB Write: %6.1fGB",
		ctx.memLimits.Total,
		ctx.memLimits.Available,
		float64(ctx.memLimits.NumWorkers))

	logger.LogStatus(ctx.localLog, constants.LogMem,
		"Block: %6.1fGB Write: %6.1fGB Comp: %7.1fGB",
		float64(ctx.memLimits.BlockCache)/(1024*1024*1024),
		float64(ctx.memLimits.WriteBuffer)/(1024*1024*1024),
		float64(ctx.memLimits.CompactionSize)/(1024*1024*1024))

	logger.LogStatus(ctx.localLog, constants.LogMem,
		"Batch: %7dk Chan: %8dk RNG: %9dk",
		ctx.memLimits.BatchSize/1000,
		ctx.memLimits.ChannelBuffer/1000,
		ctx.memLimits.RNGPoolSize/1000)

	// ctx.addressChan = make(chan *WalletInfo, ctx.memLimits.ChannelBuffer)
	ctx.doneChan = make(chan struct{})
	batchSize := ctx.memLimits.BatchSize

	ctx.parser.VerifyStats()
	ctx.blockLoader.LogAddressSummary("Block Processing Complete", true)

	// Start the generator goroutine
	startGenerator(ctx, batchSize)
	// Start the address checker goroutine
	startAddressChecker(ctx)
	// Start memory monitoring
	startMemoryMonitoring(ctx)

	return nil
}

// startGenerator handles generating addresses and managing resources.
func startGenerator(ctx *AppContext, batchSize int) {
	logger.LogStatus(ctx.localLog, constants.LogInfo,
		"Starting generator in GPU %s mode",
		utils.BoolToEnabledDisabled(ctx.gpuMode))

	ctx.addressChan = make(chan *WalletInfo, constants.AddressCheckerBatchSize*8)

	// Stats logging goroutine
	ctx.wg.Add(1) // Add wait group for stats goroutine
	go func() {
		defer ctx.wg.Done()

		statsTicker := time.NewTicker(constants.ImportLogInterval)
		defer statsTicker.Stop() // Move defer inside the goroutine

		lastGenCount := uint64(0)
		startTime := time.Now()
		lastStatsTime := time.Now()

		for {
			select {
			case <-ctx.shutdownChan:
				return
			case <-statsTicker.C:
				now := time.Now()
				elapsed := now.Sub(lastStatsTime)

				constants.GlobalStats.RLock()
				currentGenCount := constants.GlobalStats.Generated
				constants.GlobalStats.RUnlock()

				generatedDelta := currentGenCount - lastGenCount
				generatedRate := float64(generatedDelta) / elapsed.Seconds()

				var m runtime.MemStats
				runtime.ReadMemStats(&m)

				logger.LogGeneratorStats(ctx.localLog,
					float64(currentGenCount)/1e6,
					generatedRate,
					time.Since(startTime),
					float64(m.Alloc)/(1<<30),
					constants.GlobalStats.Found,
					float64(constants.GlobalStats.LegacyCount),
					float64(constants.GlobalStats.SegwitCount),
					float64(constants.GlobalStats.NativeCount))

				lastGenCount = currentGenCount
				lastStatsTime = now
			}
		}
	}()

	if ctx.gpuMode && ctx.gpuInfo.Available && ctx.gpuInfo.UsingCUDA {
		ctx.wg.Add(1)
		go func() {
			defer ctx.wg.Done()
			defer ctx.generator.Close()

			gpuBatchSize := batchSize

			for {
				select {
				case <-ctx.shutdownChan:
					return
				case <-ctx.doneChan:
					return
				default:
					checkMemoryUsage(ctx)

					privateKeys, addresses, addrTypes, err := ctx.generator.Generate(gpuBatchSize)
					if err != nil {
						logger.LogError(ctx.localLog, constants.LogError, err, "Address generation failed")
						time.Sleep(100 * time.Millisecond)
						continue
					}

					// Process in larger chunks (25% of the batch)
					chunkSize := len(addresses) / 4
					for i := 0; i < len(addresses); i += chunkSize {
						end := i + chunkSize
						if end > len(addresses) {
							end = len(addresses)
						}

						// Send chunk to channel without retries
						for j := i; j < end; j++ {
							if addresses[j] != "" && privateKeys[j] != nil {
								select {
								case ctx.addressChan <- &WalletInfo{
									Address:    addresses[j],
									PrivateKey: privateKeys[j],
									AddrType:   addrTypes[j],
								}:
								default:
									constants.TotalDroppedAddresses.Add(1)
								}
							}
						}
					}
				}
			}
		}()
	}

	if ctx.cpuMode || (!ctx.gpuMode && !ctx.gpuInfo.Available) {
		ctx.wg.Add(1) // CPU goroutine
		go func() {
			defer ctx.wg.Done()
			cpuGenerator := gpu.NewGenerator(false)
			defer cpuGenerator.Close()

			for {
				select {
				case <-ctx.shutdownChan:
					return
				case <-ctx.doneChan:
					return
				default:
					checkMemoryUsage(ctx)

					privateKeys, addresses, addrTypes, err := cpuGenerator.Generate(batchSize)
					if err != nil {
						logger.LogError(ctx.localLog,
							constants.LogError,
							err,
							"Address generation failed")
						time.Sleep(100 * time.Millisecond)
						continue
					}

					for i := range addresses {
						if addresses[i] != "" && privateKeys[i] != nil {
							sendToChecker(ctx, addresses[i], privateKeys[i], addrTypes[i])
						}
					}
				}
			}
		}()
	}
}

func testAddressMatching(ctx *AppContext) error {
	numKeys := constants.GPUTestAddresses
	startTime := time.Now()
	matched := 0

	// Calculate progress intervals
	interval := numKeys / 10 // 10% intervals
	lastProgress := 0

	// Get all addresses into a slice for random selection
	allAddresses := ctx.parser.GetAddresses()

	logger.LogHeaderStatus(ctx.localLog, constants.LogCheck,
		"Testing Matching against %s addresses",
		utils.FormatWithCommas(len(allAddresses)))
	logger.PrintSeparator(constants.LogCheck)

	for i := 0; i < numKeys; i++ {
		// Pick a random existing address
		randomIndex := rand.Intn(len(allAddresses))
		testAddr := allAddresses[randomIndex]

		if exists, _ := ctx.parser.CheckAddress(testAddr); exists {
			matched++
		}

		// Show progress every 10%
		if i > 0 && i%interval == 0 && i/interval > lastProgress {
			progress := (i * 100) / numKeys
			currentRate := float64(i) / time.Since(startTime).Seconds()
			logger.LogStatus(ctx.localLog, constants.LogCheck,
				"Progress: %d%% (%s/sec)",
				progress,
				utils.FormatWithCommas(int(currentRate)))
			lastProgress = i / interval
		}
	}

	duration := time.Since(startTime)
	rate := float64(numKeys) / duration.Seconds()

	logger.LogHeaderStatus(ctx.localLog, constants.LogInfo,
		"Checked %s addresses in %.3f seconds",
		utils.FormatWithCommas(numKeys),
		duration.Seconds())
	logger.LogStatus(ctx.localLog, constants.LogInfo,
		"Matching rate: %s addresses/second",
		utils.FormatWithCommas(int(rate)))
	if matched > 0 {
		logger.LogStatus(ctx.localLog, constants.LogInfo,
			"Found %s matching addresses!",
			utils.FormatWithCommas(matched))
	}
	logger.PrintSeparator(constants.LogInfo)

	return nil
}

// Works with `startAddressChecker`
// Periodic logging of System Memory usage
func startMemoryMonitoring(ctx *AppContext) {
	memTicker := time.NewTicker(constants.ImportLogInterval * 30)
	defer memTicker.Stop()

	for {
		select {
		case <-memTicker.C:
			v, _ := mem.VirtualMemory()
			logger.LogHeaderStatus(ctx.localLog, constants.LogMem,
				"Grendel Memory Used %.1fGB, Available: %.1fGB",
				float64(v.Used)/(1024*1024*1024),      // Convert used memory to GB
				float64(v.Available)/(1024*1024*1024)) // Convert available memory to GB
			logger.PrintSeparator(constants.LogMem)
		case <-ctx.shutdownChan:
			return
		}
	}
}
