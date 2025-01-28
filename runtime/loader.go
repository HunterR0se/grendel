package runtime

import (
	"Grendel/constants"
	"Grendel/generator"
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

	"github.com/shirou/gopsutil/mem"
)

func Initialize(genMode, debugMode, forceReparse bool, gpuInfo gpu.GPUInfo, cpuMode, gpuMode bool) (*AppContext, gpu.Generator, error) {
	ctx := &AppContext{
		LocalLog:     log.New(os.Stdout, "", 0),
		ShutdownChan: make(chan struct{}),
		GenMode:      genMode,
		DebugMode:    debugMode,
		ForceReparse: forceReparse,
		DoneChan:     make(chan struct{}), // shutdown
		Wg:           sync.WaitGroup{},    // waitgroup
		GpuInfo:      gpuInfo,             // for GPU
		CpuMode:      cpuMode,
		GpuMode:      gpuMode, // Add this line
	}

	// Set up logger constant
	constants.Logger = log.New(os.Stdout, "", log.LstdFlags)
	constants.GeneratorMode = ctx.GenMode

	if ctx.DebugMode {
		constants.DebugMode = true
	}

	// Check GPU availability
	logSystemInfo(ctx)

	// Setup paths
	ctx.BaseDir = utils.GetBaseDir()
	ctx.ParserPath = filepath.Join(ctx.BaseDir, ".bitcoin")
	ctx.DbPath = filepath.Join(ctx.BaseDir, constants.AddressDBPath)

	// Create bitcoin directory if needed
	if err := createBitcoinDir(ctx.BaseDir); err != nil {
		return nil, nil, fmt.Errorf("failed to create bitcoin directory: %w", err)
	}

	// Initialize parser
	newParser, err := parser.NewParser(ctx.LocalLog, ctx.ParserPath, ctx.DbPath, ctx.ForceReparse)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to initialize parser: %w", err)
	}
	ctx.Parser = newParser

	// Initialize parser stats
	ctx.Parser.Stats = &constants.AddressCategory{}

	// Verify database
	if err := verifyDatabase(ctx.Parser); err != nil {
		return nil, nil, fmt.Errorf("database verification failed: %w", err)
	}

	// Set up shutdown handler
	ctx.ShutdownChan = SetupGracefulShutdown(ctx.LocalLog, ctx.ShutdownChan, ctx.DoneChan)

	// Initialize generator
	var generatorInstance gpu.Generator
	if ctx.CpuMode {
		generatorInstance, err = gpu.NewCPUGenerator(nil)
	} else if ctx.GpuMode && ctx.GpuInfo.Available && ctx.GpuInfo.UsingCUDA {
		generatorInstance, err = gpu.NewCUDAGenerator(nil)
	} else {
		generatorInstance, err = gpu.NewCPUGenerator(nil) // Default to CPU
	}
	if err != nil {
		return nil, nil, fmt.Errorf("failed to initialize generator: %v", err)
	}

	ctx.Generator = generatorInstance

	return ctx, generatorInstance, nil
}

// 2. Load blocks
func LoadBlocks(ctx *AppContext) error {
	logger.LogHeaderStatus(ctx.LocalLog, constants.LogDB, "Starting the Block Loader...")
	logger.PrintSeparator(constants.LogDB)

	ctx.BlockLoader = loader.NewBlockLoader(ctx.Parser)

	blockFiles, err := ctx.BlockLoader.FindBlockFiles()
	if err != nil {
		return fmt.Errorf("failed to scan for blocks: %w", err)
	}

	if len(blockFiles) == 0 {
		logger.LogStatus(ctx.LocalLog, constants.LogDB, "No new blocks found to process")
		return StartBackgroundLoader(ctx)
	}

	logger.LogDebug(ctx.LocalLog, constants.LogDB,
		"Found %d blocks to process", len(blockFiles))

	if err := ctx.BlockLoader.LoadNewBlocks(); err != nil {
		logger.LogError(ctx.LocalLog, constants.LogError, err,
			"Block loading encountered errors but continuing...")
	}

	if !constants.AreAddressesLoaded {
		count, err := ctx.Parser.LoadAllAddresses()
		if err != nil {
			logger.LogError(ctx.LocalLog, constants.LogError, err,
				"Failed to load addresses from database")
		} else {
			logger.LogStatus(ctx.LocalLog, constants.LogDB,
				"Loaded %s addresses from database", utils.FormatWithCommas(int(count)))
			ctx.AddressesLoaded = true
		}

		ctx.BlockLoader.LogAddressSummary("Block Processing Complete", true)
	}

	logger.LogStatus(ctx.LocalLog, constants.LogDB, "Block loader successfully started.")
	return StartBackgroundLoader(ctx)
}

func StartBackgroundLoader(ctx *AppContext) error {
	// Create a map to track processed blocks using their base names
	processedBlocks := make(map[string]bool)

	// Record initially processed blocks by their base names
	initialFiles, err := ctx.BlockLoader.FindBlockFiles()
	if err == nil {
		for _, file := range initialFiles {
			baseName := filepath.Base(file)
			if ctx.Parser.IsBlockProcessed(file) {
				processedBlocks[baseName] = true
				if constants.DebugMode {
					logger.LogDebug(ctx.LocalLog, constants.LogDB,
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
			case <-ctx.ShutdownChan:
				return
			case <-ticker.C:
				// Only scan if enough time has passed
				if time.Since(lastScan) < 4*time.Minute {
					continue
				}

				// Find any new blocks
				currentFiles, err := ctx.BlockLoader.FindBlockFiles()
				if err != nil {
					logger.LogError(ctx.LocalLog, constants.LogError, err,
						"Failed to scan for new blocks")
					continue
				}

				// Only process blocks we haven't seen before AND aren't processed
				var newBlocks []string
				for _, file := range currentFiles {
					baseName := filepath.Base(file)
					if !processedBlocks[baseName] {
						if !ctx.Parser.IsBlockProcessed(file) {
							newBlocks = append(newBlocks, file)
							processedBlocks[baseName] = true
							if constants.DebugMode {
								logger.LogDebug(ctx.LocalLog, constants.LogDB,
									"Found new block to process: %s", baseName)
							}
						}
					}
				}

				if len(newBlocks) > 0 {
					logger.LogDebug(ctx.LocalLog, constants.LogDB,
						"Processing %d new blocks", len(newBlocks))

					// Process only the new blocks directly
					err = ctx.BlockLoader.ProcessNewBlocks(
						newBlocks,
						ctx.Parser.BlocksProcessed,
						ctx.Parser.AddressesFound)

					if err != nil {
						logger.LogError(ctx.LocalLog, constants.LogError, err,
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
func StartAddressGeneration(ctx *AppContext) error {
	if ctx.Generator == nil {
		return fmt.Errorf("generator not initialized")
	}

	ctx.MemLimits = utils.CalculateMemoryLimits()

	availDisk, totalDisk := utils.GetAvailableDiskSpace()
	logger.LogHeaderStatus(ctx.LocalLog, constants.LogMem,
		"CPU Cores: %4d Disk: %7.1fGB Free:  %6.1fGB",
		runtime.NumCPU(),
		totalDisk, availDisk)

	logger.LogStatus(ctx.LocalLog, constants.LogMem,
		"Total: %6.1fGB Avail: %6.1fGB Write: %6.1fGB",
		ctx.MemLimits.Total,
		ctx.MemLimits.Available,
		float64(ctx.MemLimits.NumWorkers))

	logger.LogStatus(ctx.LocalLog, constants.LogMem,
		"Block: %6.1fGB Write: %6.1fGB Comp: %7.1fGB",
		float64(ctx.MemLimits.BlockCache)/(1024*1024*1024),
		float64(ctx.MemLimits.WriteBuffer)/(1024*1024*1024),
		float64(ctx.MemLimits.CompactionSize)/(1024*1024*1024))

	logger.LogStatus(ctx.LocalLog, constants.LogMem,
		"Batch: %7dk Chan: %8dk RNG: %9dk",
		ctx.MemLimits.BatchSize/1000,
		ctx.MemLimits.ChannelBuffer/1000,
		ctx.MemLimits.RNGPoolSize/1000)

	// ctx.addressChan = make(chan *WalletInfo, ctx.memLimits.ChannelBuffer)
	ctx.DoneChan = make(chan struct{})
	batchSize := ctx.MemLimits.BatchSize

	ctx.Parser.VerifyStats()
	ctx.BlockLoader.LogAddressSummary("Block Processing Complete", true)

	// Start the generator goroutine
	StartGenerator(ctx, batchSize)
	// Start the address checker goroutine
	StartAddressChecker(ctx)
	// Start memory monitoring
	StartMemoryMonitoring(ctx)

	return nil
}

// startGenerator handles generating addresses and managing resources.
func StartGenerator(ctx *AppContext, batchSize int) {
	logger.LogStatus(ctx.LocalLog, constants.LogInfo,
		"Starting generator in GPU %s mode",
		utils.BoolToEnabledDisabled(ctx.GpuMode))

	ctx.AddressChan = make(chan *WalletInfo, constants.AddressCheckerBatchSize*8)

	// Stats logging goroutine
	ctx.Wg.Add(1) // Add wait group for stats goroutine
	go func() {
		defer ctx.Wg.Done()

		statsTicker := time.NewTicker(constants.ImportLogInterval)
		defer statsTicker.Stop() // Move defer inside the goroutine

		lastGenCount := uint64(0)
		startTime := time.Now()
		lastStatsTime := time.Now()

		for {
			select {
			case <-ctx.ShutdownChan:
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

				logger.LogGeneratorStats(ctx.LocalLog,
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

	if ctx.GpuMode && ctx.GpuInfo.Available && ctx.GpuInfo.UsingCUDA {
		ctx.Wg.Add(1)
		go func() {
			defer ctx.Wg.Done()
			defer ctx.Generator.Close()

			gpuBatchSize := batchSize

			for {
				select {
				case <-ctx.ShutdownChan:
					return
				case <-ctx.DoneChan:
					return
				default:
					CheckMemoryUsage(ctx)

					privateKeys, addresses, addrTypes, err := ctx.Generator.Generate(gpuBatchSize)
					if err != nil {
						logger.LogError(ctx.LocalLog, constants.LogError, err, "Address generation failed")
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
								case ctx.AddressChan <- &WalletInfo{
									Address:    addresses[j],
									PrivateKey: privateKeys[j],
									AddrType:   addrTypes[j],
								}:
								default:
									constants.TotalDroppedAddresses.Add(1)
									// Add this line:
									generator.DecrementAddressType(addrTypes[j])
								}
							}
						}
					}
				}
			}
		}()
	}

	if ctx.CpuMode || (!ctx.GpuMode && !ctx.GpuInfo.Available) {
		ctx.Wg.Add(1) // CPU goroutine
		go func() {
			defer ctx.Wg.Done()
			cpuGenerator := gpu.NewGenerator(false)
			defer cpuGenerator.Close()

			for {
				select {
				case <-ctx.ShutdownChan:
					return
				case <-ctx.DoneChan:
					return
				default:
					CheckMemoryUsage(ctx)

					privateKeys, addresses, addrTypes, err := cpuGenerator.Generate(batchSize)
					if err != nil {
						logger.LogError(ctx.LocalLog,
							constants.LogError,
							err,
							"Address generation failed")
						time.Sleep(100 * time.Millisecond)
						continue
					}

					for i := range addresses {
						if addresses[i] != "" && privateKeys[i] != nil {
							SendToChecker(ctx, addresses[i], privateKeys[i], addrTypes[i])
						}
					}
				}
			}
		}()
	}
}

func TestAddressMatching(ctx *AppContext) error {
	matched, duration, rate, err := CheckAddresses(ctx, constants.GPUTestAddresses)
	if err != nil {
		return err
	}

	ReportAddressCheckResults(ctx, constants.GPUTestAddresses, matched, duration, rate)
	return nil
}

// Works with `startAddressChecker`
// Periodic logging of System Memory usage
func StartMemoryMonitoring(ctx *AppContext) {
	memTicker := time.NewTicker(constants.ImportLogInterval * 30)
	defer memTicker.Stop()

	for {
		select {
		case <-memTicker.C:
			v, _ := mem.VirtualMemory()
			logger.LogHeaderStatus(ctx.LocalLog, constants.LogMem,
				"Grendel Memory Used %.1fGB, Available: %.1fGB",
				float64(v.Used)/(1024*1024*1024),      // Convert used memory to GB
				float64(v.Available)/(1024*1024*1024)) // Convert available memory to GB
			logger.PrintSeparator(constants.LogMem)
		case <-ctx.ShutdownChan:
			return
		}
	}
}
