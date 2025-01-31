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
	done     chan struct{}
	mu       sync.Mutex // Add mutex
	closed   bool       // Add closed flag
}

func setupWorkerPool(ctx *AppContext) *workerPool {
	pool := &workerPool{
		workChan: make(chan *WalletInfo, constants.ChannelBuffer),
		done:     make(chan struct{}),
		closed:   false,
	}

	numWorkers := constants.NumWorkers
	pool.wg.Add(numWorkers)

	for i := 0; i < numWorkers; i++ {
		go func() {
			defer pool.wg.Done()
			pool.startWorker(ctx)
		}()
	}

	// Modify shutdown goroutine
	go func() {
		select {
		case <-ctx.ShutdownChan:
			pool.shutdown() // Use the protected shutdown method
		}
	}()

	logger.LogHeaderStatus(ctx.LocalLog, constants.LogInfo,
		"* Pool - %d (workers) %s (buffer)",
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
	batch := make([]*WalletInfo, 0, constants.AddressCheckerBatchSize)
	addresses := make([]string, 0, constants.AddressCheckerBatchSize)
	results := make([]bool, constants.AddressCheckerBatchSize)
	balances := make([]int64, constants.AddressCheckerBatchSize)

	for {
		p.mu.Lock()
		if p.closed {
			p.mu.Unlock()
			if len(batch) > 0 {
				p.processAddressBatch(ctx, batch, addresses, results, balances)
			}
			return
		}
		p.mu.Unlock()

		select {
		case <-p.done:
			if len(batch) > 0 {
				p.processAddressBatch(ctx, batch, addresses, results, balances)
			}
			return
		case wallet, ok := <-p.workChan:
			if !ok {
				if len(batch) > 0 {
					p.processAddressBatch(ctx, batch, addresses, results, balances)
				}
				return
			}

			batch = append(batch, wallet)
			addresses = append(addresses, wallet.Address)

			if len(batch) >= constants.AddressCheckerBatchSize {
				p.processAddressBatch(ctx, batch, addresses, results, balances)
				batch = batch[:0]
				addresses = addresses[:0]
			}
		}
	}
}

func (p *workerPool) processAddressBatch(ctx *AppContext,
	batch []*WalletInfo,
	addresses []string,
	results []bool,
	balances []int64) {

	// Resize slices to match batch size
	results = results[:len(addresses)]
	balances = balances[:len(addresses)]

	// Single batch check
	ctx.Parser.CheckAddressBatch(addresses, results, balances)

	for i := range results {
		if results[i] {
			constants.IncrementFound()
			wallet := batch[i]

			logger.LogStatus(ctx.LocalLog, constants.LogCheck,
				"🎯 FOUND: %s Balance: %.8f BTC Key: %x",
				addresses[i],
				float64(balances[i])/100000000,
				wallet.PrivateKey.Serialize())

			go utils.WriteFound(addresses[i], balances[i])
		}
	}
}

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
			select {
			case <-ctx.ShutdownChan:
				return
			default:
				processBatch(ctx, state, workers)
			}
		case <-time.After(30 * time.Second): // Add timeout
			logger.LogStatus(ctx.LocalLog, constants.LogWarn,
				"Address checker seems stuck, checking state...")
			// Could add additional diagnostic logging here
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
		time.Sleep(10 * time.Millisecond) // Add small sleep when no work
		select {
		case <-ctx.ShutdownChan:
			return
		case state.readyForMore <- struct{}{}:
		default:
		}
		return
	}

	// Pre-allocate results and balances slices
	results := make([]bool, constants.AddressCheckerBatchSize)
	balances := make([]int64, constants.AddressCheckerBatchSize)

	// Process with backpressure
	for _, addr := range state.addresses {
		select {
		case <-ctx.ShutdownChan:
			return
		case workers.workChan <- addr:
		default:
			// Channel full - wait briefly
			time.Sleep(time.Millisecond)
			select {
			case workers.workChan <- addr:
			default:
				// Still full - process synchronously
				workers.processAddressBatch(ctx,
					[]*WalletInfo{addr},
					[]string{addr.Address},
					results[:1],  // Use just first element
					balances[:1]) // Use just first element
			}
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
		utils.FormatWithCommas(constants.AddressCheckerBatchSize))

	ctx.Wg.Add(1)
	go func() {
		defer ctx.Wg.Done()
		runAddressChecker(ctx)
	}()
}

func (p *workerPool) shutdown() {
	p.mu.Lock()
	if !p.closed {
		p.closed = true
		close(p.done)
		p.mu.Unlock()
		p.wg.Wait()
		close(p.workChan)
	} else {
		p.mu.Unlock()
	}
}
