package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"Grendel/constants"
	"Grendel/generator"
	"Grendel/gpu"
	"Grendel/loader"
	"Grendel/logger"
	"Grendel/parser"
	"Grendel/utils"

	"github.com/btcsuite/btcd/btcec/v2"

	// pprof -- flag of --profile
	"net/http"
	_ "net/http/pprof"
)

var (
	localLog      *log.Logger
	cachedBaseDir string
	baseDirOnce   sync.Once
	lastStatsTime = time.Now()
)

type WalletInfo struct {
	Address    string
	PrivateKey *btcec.PrivateKey
	AddrType   generator.AddressType
	// Seed       string // Uncomment if/when we add seed phrase support
}

var closeOnce sync.Once

type AppContext struct {
	localLog     *log.Logger
	genMode      bool
	debugMode    bool
	forceReparse bool
	cpuMode      bool
	gpuMode      bool // Add this line
	baseDir      string
	parserPath   string
	dbPath       string
	addressPath  string
	parser       *parser.Parser
	shutdownChan chan struct{}

	addressChan     chan *WalletInfo
	doneChan        chan struct{}
	memLimits       *utils.MemoryLimits
	blockLoader     *loader.BlockLoader
	wg              sync.WaitGroup
	closeOnce       sync.Once
	blockLoadDone   chan struct{}
	addressesLoaded bool
	// GPU!
	gpuInfo   gpu.GPUInfo
	generator gpu.Generator
}

func main() {
	// Initialize localLog first
	localLog = log.New(os.Stdout, "", 0)
	var err error

	// Check for GPU ONCE at the start
	gpuInfo := gpu.CheckForGPU()

	// Parse flags first, passing gpuInfo
	genMode, debugMode, forceReparse, trackAll, importMode, cpuMode, gpuMode, extractMode := setupFlags(gpuInfo)
	constants.GeneratorMode = *genMode
	constants.TrackAllAddresses = *trackAll

	// Set debug mode early if enabled
	if *debugMode {
		constants.DebugMode = true
		logger.LogStatus(localLog, constants.LogDebug, "Debug mode enabled")
	}

	// Handle import mode
	if *importMode {
		logger.LogStatus(localLog, constants.LogInfo, "* Re-import all addresses! *")
		*forceReparse = true // Force reparse when importing
	}

	fmt.Print("\033[H\033[2J\n")
	logger.Banner()

	// Check arguments after parsing flags
	if len(os.Args) < 2 {
		flag.Usage()
		os.Exit(1)
	}

	// IMPORTER
	if *importMode {
		if err := handleImport(localLog); err != nil {
			logger.LogError(localLog, constants.LogError, err, "Import failed")
			os.Exit(1)
		}
		os.Exit(0)
	}

	// EXTRACTOR
	if *extractMode {
		if err := parser.ExtractAddresses(); err != nil {
			logger.LogError(localLog, constants.LogError, err, "Import failed")
			os.Exit(1)
		}
		os.Exit(0)
	}

	// Show all our settings with a clean output
	logger.LogHeaderStatus(localLog, constants.LogInfo,
		"Generator: %-10v Track All:  %-10v",
		utils.BoolToEnabledDisabled(*genMode),
		utils.BoolToEnabledDisabled(*trackAll))
	logger.LogStatus(localLog, constants.LogInfo,
		"Reparse:   %-10v GPU Mode:   %-10v",
		utils.BoolToEnabledDisabled(*forceReparse),
		utils.BoolToEnabledDisabled(*gpuMode))
	logger.LogStatus(localLog, constants.LogInfo,
		"DebugMode: %-10v CPU Mode:   %-10v",
		utils.BoolToEnabledDisabled(*debugMode),
		utils.BoolToEnabledDisabled(*cpuMode))

	// If GPU is available and we're using GPU mode, run CUDA test
	if gpuInfo.Available && *gpuMode {

		// Create temporary CUDA generator just for testing
		testGen, err := gpu.NewCUDAGenerator(&generator.Config{})
		if err != nil {
			logger.LogError(localLog, constants.LogError, err, "Failed to create CUDA test generator")
			os.Exit(1)
		}
		defer testGen.Close()

		if err := testGen.TestGeneration(); err != nil {
			logger.LogError(localLog, constants.LogError, err, "CUDA test failed")
			os.Exit(1)
		}

	}

	var wg sync.WaitGroup

	// If forceReparse is true, delete the existing database
	if *forceReparse {
		dbPath := filepath.Join(getBaseDir(), constants.AddressDBPath)
		if err := os.RemoveAll(dbPath); err != nil {
			logger.LogError(localLog, constants.LogError, err, "Failed to delete existing database")
			logger.PrintSeparator(constants.LogError)
			time.Sleep(100 * time.Millisecond)
			os.Exit(1)
		}
		logger.LogStatus(localLog, constants.LogDB, "Removed existing database")
	}

	ctx, generatorInstance, err := initialize(*genMode,
		*debugMode, *forceReparse,
		gpuInfo, *cpuMode, *gpuMode)

	if err != nil {
		if generatorInstance != nil {
			generatorInstance.Close()
		}
		logger.LogError(localLog, constants.LogError, err, "Initialization error")
		os.Exit(1)
	}

	defer ctx.parser.Cleanup()

	// Start Garbage Collection Thread
	wg.Add(1)
	go func() {
		defer wg.Done()
		utils.PeriodicGC()
	}()

	// Load blocks - continue even if there are errors
	if err := loadBlocks(ctx); err != nil {
		logger.LogError(localLog, constants.LogError, err,
			"Block loading encountered errors...")
	}

	// Ensure we have loaded addresses before starting generation
	if !ctx.addressesLoaded {
		count, err := ctx.parser.LoadAllAddresses()
		if err != nil {
			logger.LogError(localLog, constants.LogError, err,
				"Failed to load addresses but continuing...")
		} else {
			logger.LogStatus(localLog, constants.LogInfo,
				"Loaded %s addresses (from Blockchain)",
				utils.FormatWithCommas(int(count)))
			ctx.addressesLoaded = true
		}
	}

	constants.ObfuscationKey, err = utils.ObfuscationKey()
	if err != nil {
		logger.LogHeaderStatus(localLog, constants.LogInfo,
			"Could Not Find Obfuscation Key")
		logger.LogStatus(localLog, constants.LogInfo,
			"* If you are running a pruned node, the obfuscation")
		logger.LogStatus(localLog, constants.LogInfo,
			"  key may not be available. Continuing without it...")
		logger.PrintSeparator(constants.LogInfo)
		constants.ObfuscationKey = nil
	}

	// load the stats for the logger, so we have the correct amounts.
	logger.FirstLoad()

	// Test address matching
	if err := testAddressMatching(ctx); err != nil {
		logger.LogError(localLog, constants.LogError, err, "Address matching test failed")
		os.Exit(1)
	}

	// Start address generation if enabled
	if ctx.genMode {
		if err := startAddressGeneration(ctx); err != nil {
			logger.LogError(localLog, constants.LogError, err,
				"Address generation initialization failed")
			os.Exit(1)
		}
	}

	// Run main application loop
	runMainLoop(ctx)

	// wait for finish
	wg.Wait()
}

func runMainLoop(ctx *AppContext) {
	newTicker := time.NewTicker(constants.ImportLogInterval)
	defer newTicker.Stop()

	// Create a function to handle cleanup
	cleanup := func() {
		closeOnce.Do(func() {
			// First signal all goroutines to stop
			close(ctx.shutdownChan)

			// Signal processing should stop
			close(ctx.doneChan)

			// Wait for all goroutines to finish
			ctx.wg.Wait()

			// Only then close the address channel
			if ctx.addressChan != nil {
				close(ctx.addressChan)
			}

			ctx.parser.Cleanup()
		})
	}

	for {
		select {
		case <-newTicker.C:
			// Log block processing status only
			logger.LogDebug(ctx.localLog, constants.LogStats,
				"Blocks Processed: %d, Addresses Found: %d",
				ctx.parser.BlocksProcessed,
				ctx.parser.AddressesFound)

		case <-ctx.shutdownChan:
			cleanup()
			logger.LogStatus(ctx.localLog, constants.LogWarn, "System Shutdown Complete.")
			return
		}
	}
}

// setupGracefulShutdown initializes graceful shutdown handling for the application.
// It creates channels for shutdown signaling and sets up signal handling for
// SIGTERM and SIGINT. When a shutdown signal is received, it cleans up resources
// and exits cleanly.
//
// Parameters:
//   - localLog: Logger instance for status messages
//
// Returns:
//   - chan struct{}: Channel that will be closed on shutdown signal
func setupGracefulShutdown(localLog *log.Logger,
	shutdownChan,
	doneChan chan struct{}) chan struct{} {

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGTERM, syscall.SIGINT, syscall.SIGQUIT)

	go func() {
		sig := <-sigChan
		signal.Stop(sigChan)
		close(sigChan)

		fmt.Print("\r\033[K")
		logger.LogHeaderStatus(localLog, constants.LogWarn,
			"Received signal %v, initiating shutdown...", sig)
		logger.PrintSeparator(constants.LogWarn)

		// Use closeOnce to ensure channels are closed only once
		closeOnce.Do(func() {
			close(shutdownChan)
			close(doneChan) // Close doneChan to make goroutines exit immediately
		})
	}()

	return shutdownChan
}

func setupFlags(gpuInfo gpu.GPUInfo) (*bool,
	*bool, *bool, *bool, *bool, *bool, *bool, *bool) {
	genMode := flag.Bool("gen", true, "Generate new addresses (default)")
	debugMode := flag.Bool("debug", false, "Enable debug mode")
	forceReparse := flag.Bool("force", false, "Force reparse of all blocks")
	trackAll := flag.Bool("track-all", true, "Track all addresses (default)")
	extractMode := flag.Bool("extract", false, "Extract all addresses from db")
	importMode := flag.Bool("import", false, "Re-import all addresses")
	profiling := flag.Bool("profile", false, "Turn on PProf (port 6060)")
	cpuMode := flag.Bool("cpu", false, "Force CPU mode for testing")
	gpuMode := flag.Bool("gpu", false, "Force GPU mode for testing")
	benchMode := flag.Bool("bench", false, "Run benchmark mode")

	flag.Usage = func() {
		logger.PrintSeparator(constants.LogStart)
		localLog.Printf("%s %s Grendel Commands:",
			constants.LogStart,
			constants.EmojiBitcoin)
		localLog.Printf("%s --debug     : Enable Debug Mode", constants.LogStart)
		localLog.Printf("%s --gen       : Generate Addresses (true by default)", constants.LogStart)
		localLog.Printf("%s --import    : Re-import all addresses", constants.LogStart)
		localLog.Printf("%s --track-all : Track all addresses (uses more memory)", constants.LogStart)
		localLog.Printf("%s --cpu       : Force CPU mode for testing", constants.LogStart)
		localLog.Printf("%s --gpu       : Force GPU mode for testing", constants.LogStart)
		localLog.Printf("%s --bench     : Run Benchmark and exit", constants.LogStart)
		logger.PrintSeparator(constants.LogStart)
	}

	flag.Parse()

	if *benchMode {
		benchmark()
		os.Exit(1)
	}

	if *profiling {
		go func() {
			log.Println(http.ListenAndServe("0.0.0.0:6060", nil))
		}()
	}

	// Automatically check for GPU if neither CPU nor GPU mode is explicitly specified
	if !*cpuMode && !*gpuMode {
		if gpuInfo.Available && gpuInfo.UsingCUDA {
			*gpuMode = true
		} else {
			*cpuMode = true
		}
	}

	return genMode, debugMode, forceReparse,
		trackAll, importMode, cpuMode, gpuMode, extractMode
}
