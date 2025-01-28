package runtime

import (
	"Grendel/constants"
	"Grendel/logger"
	"Grendel/utils"
	"sync"
	"time"
)

type checkerState struct {
	addresses        []*WalletInfo
	lastProgressTime time.Time
	checksCompleted  int
	startTime        time.Time
	ticker           *time.Ticker
	readyForMore     chan struct{}
}

type workerPool struct {
	workChan chan *WalletInfo
	wg       sync.WaitGroup
}

func setupWorkerPool(ctx *AppContext) *workerPool {
	pool := &workerPool{
		// Use constants for buffer size
		workChan: make(chan *WalletInfo, constants.ChannelBuffer),
	}

	// Use the constant for number of workers
	numWorkers := constants.NumWorkers
	pool.wg.Add(numWorkers)

	for i := 0; i < numWorkers; i++ {
		go pool.startWorker(ctx)
	}

	logger.LogStatus(ctx.LocalLog, constants.LogInfo,
		"Worker Pool - %d (workers) %s (buffer)",
		numWorkers,
		utils.FormatWithCommas(constants.ChannelBuffer))

	return pool
}

var addressBatchPool = sync.Pool{
	New: func() interface{} {
		return make([]string, 0, constants.AddressCheckerBatchSize)
	},
}

func (p *workerPool) startWorker(ctx *AppContext) {
	defer p.wg.Done()

	batch := make([]*WalletInfo, 0, constants.AddressCheckerBatchSize)
	addresses := make([]string, 0, constants.AddressCheckerBatchSize)

	for addr := range p.workChan {
		batch = append(batch, addr)
		addresses = append(addresses, addr.Address)

		// Process when batch is full
		if len(batch) >= constants.AddressCheckerBatchSize {
			p.processAddressBatch(ctx, batch, addresses)
			batch = batch[:0]
			addresses = addresses[:0]
		}
	}

	// Process any remaining addresses
	if len(batch) > 0 {
		p.processAddressBatch(ctx, batch, addresses)
	}
}

func (p *workerPool) processAddressBatch(ctx *AppContext, batch []*WalletInfo, addresses []string) {
	// Pre-allocate all slices at once
	results := make([]bool, len(addresses))
	balances := make([]int64, len(addresses))

	// Single batch check
	ctx.Parser.CheckAddressBatch(addresses, results, balances)

	for i := range results {
		if results[i] {
			constants.IncrementFound()
			wallet := batch[i]

			// Log found address details
			logger.LogStatus(ctx.LocalLog, constants.LogCheck,
				"🎯 FOUND: %s Balance: %.8f BTC Key: %x",
				addresses[i],
				float64(balances[i])/100000000,
				wallet.PrivateKey.Serialize())

			// Write to file asynchronously
			go utils.WriteFound(addresses[i], balances[i])
		}
	}

}

// -------- HELPER FUNCTIONS AND ADDITIONAL FUNCTIONS ---------------

func newCheckerState() *checkerState {
	state := &checkerState{
		addresses:        make([]*WalletInfo, 0, constants.ImportBatchSize),
		startTime:        time.Now(),
		lastProgressTime: time.Now(),
		ticker:           time.NewTicker(constants.ImportLogInterval),
		readyForMore:     make(chan struct{}, 1),
	}

	state.readyForMore <- struct{}{} // Initial signal
	return state
}

func runAddressChecker(ctx *AppContext) {
	logger.LogStatus(ctx.LocalLog, constants.LogInfo, "Block Status logger Started")
	defer logger.LogDebug(ctx.LocalLog, constants.LogInfo, "Stats logger stopped")

	state := newCheckerState()
	defer state.ticker.Stop()

	workers := setupWorkerPool(ctx)
	defer workers.shutdown()

	for {
		select {
		case <-ctx.ShutdownChan:
			return
		case <-state.ticker.C:
			logProgress(ctx, state)
		case <-state.readyForMore:
			processBatch(ctx, state, workers)
		}
	}
}

func logProgress(ctx *AppContext, state *checkerState) {
	elapsed := time.Since(state.startTime)
	rate := float64(state.checksCompleted) / elapsed.Seconds()

	logger.LogDebug(ctx.LocalLog, constants.LogInfo,
		"Checker Status: %s checks/sec, Total: %s",
		utils.FormatWithCommas(int(rate)),
		utils.FormatWithCommas(state.checksCompleted))
}

func processBatch(ctx *AppContext, state *checkerState, workers *workerPool) {
	filledCount := fillBatchAddresses(ctx, state)

	if filledCount == 0 {
		select {
		case <-ctx.ShutdownChan:
			return
		case state.readyForMore <- struct{}{}:
		default:
		}
		return
	}

	// Process entire batch at once
	for _, addr := range state.addresses {
		select {
		case <-ctx.ShutdownChan:
			return
		case workers.workChan <- addr:
		default:
			// If channel is full, process synchronously
			workers.processAddressBatch(ctx, []*WalletInfo{addr}, []string{addr.Address})
		}
	}

	// Signal ready for more immediately
	select {
	case <-ctx.ShutdownChan:
	case state.readyForMore <- struct{}{}:
	default:
	}

	state.checksCompleted += filledCount

	if time.Since(state.lastProgressTime) >= 5*time.Second {
		batchRate := float64(filledCount) / time.Since(state.lastProgressTime).Seconds()
		logger.LogDebug(ctx.LocalLog, constants.LogCheck,
			"Batch complete: %s addresses at %s/sec",
			utils.FormatWithCommas(filledCount),
			utils.FormatWithCommas(int(batchRate)))
		state.lastProgressTime = time.Now()
	}

	state.addresses = state.addresses[:0]
}

func fillBatchAddresses(ctx *AppContext, state *checkerState) int {
	filledCount := 0
	for filledCount < constants.ImportBatchSize {
		select {
		case addr, ok := <-ctx.AddressChan:
			if !ok {
				return filledCount
			}
			state.addresses = append(state.addresses, addr)
			filledCount++
		default:
			return filledCount
		}
	}
	return filledCount
}

// Primary function that gets called from sidecar.go
func StartAddressChecker(ctx *AppContext) {
	logger.LogStatus(ctx.LocalLog, constants.LogInfo,
		"Address checker started - batch size: %s",
		utils.FormatWithCommas(constants.ImportBatchSize))

	ctx.Wg.Add(1)
	go func() {
		defer ctx.Wg.Done()
		runAddressChecker(ctx)
	}()
}

func (p *workerPool) shutdown() {
	close(p.workChan)
	p.wg.Wait()
}
