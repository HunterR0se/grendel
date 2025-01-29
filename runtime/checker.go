package runtime

import (
	"Grendel/constants"
	"Grendel/logger"
	"Grendel/utils"
	"fmt"
	"runtime"
	"time"

	"golang.org/x/exp/rand"
)

// CheckAddresses checks a specified number of addresses against loaded addresses
// and reports performance and match statistics.
//
// Parameters:
//   - ctx: Application context with parser and logging
//   - numKeys: Number of addresses to check
//
// Returns:
//   - Number of matched addresses
//   - Duration of the check
//   - Rate of address checking
//   - Error if any
func CheckAddresses(ctx *AppContext, numKeys int) (int, time.Duration, float64, error) {
	startTime := time.Now()
	matched := 0

	// Get existing addresses
	allAddresses := ctx.Parser.GetAddresses()
	if len(allAddresses) == 0 {
		return 0, 0, 0, fmt.Errorf("no addresses loaded for checking")
	}

	// Pre-allocate batch arrays
	batchSize := constants.AddressCheckerBatchSize
	results := make([]bool, batchSize)
	balances := make([]int64, batchSize)
	testAddresses := make([]string, batchSize)

	// Initialize RNG for selecting addresses
	rnd := rand.New(rand.NewSource(uint64(startTime.UnixNano())))

	// Progress tracking
	interval := numKeys / 10
	lastProgress := 0

	// Process in batches
	for i := 0; i < numKeys; i += batchSize {
		currentBatchSize := min(batchSize, numKeys-i)

		// Resize slices if needed for final batch
		if currentBatchSize < batchSize {
			results = results[:currentBatchSize]
			balances = balances[:currentBatchSize]
			testAddresses = testAddresses[:currentBatchSize]
		}

		// Fill batch with addresses
		for j := 0; j < currentBatchSize; j++ {
			if j%2 == 0 { // Exactly 50% real addresses
				// Use real address
				randomIndex := rnd.Intn(len(allAddresses))
				testAddresses[j] = allAddresses[randomIndex]
			} else {
				// Generate completely new address to ensure no match
				// Format: 1 followed by 33 random hex chars
				fakeAddr := make([]byte, 34)
				fakeAddr[0] = '1' // Bitcoin address prefix
				for k := 1; k < 34; k++ {
					// Use only hex chars (0-9, a-f)
					if rnd.Float32() < 0.5 {
						fakeAddr[k] = byte(rnd.Intn(10) + '0') // 0-9
					} else {
						fakeAddr[k] = byte(rnd.Intn(6) + 'a') // a-f
					}
				}
				testAddresses[j] = string(fakeAddr)
			}
		}

		// Check batch
		ctx.Parser.CheckAddressBatch(testAddresses, results, balances)

		// Count matches
		for j := range results {
			if results[j] {
				matched++
			}
		}

		// Log progress
		if i >= interval && i/interval > lastProgress {
			currentRate := float64(i) / time.Since(startTime).Seconds()
			matchRate := float64(matched) / float64(i+currentBatchSize) * 100
			logger.LogStatus(ctx.LocalLog, constants.LogCheck,
				"Progress: %d%% (%s/sec) Match Rate: %.1f%%",
				(i*100)/numKeys,
				utils.FormatWithCommas(int(currentRate)),
				matchRate)
			lastProgress = i / interval
		}
	}

	duration := time.Since(startTime)
	rate := float64(numKeys) / duration.Seconds()

	return matched, duration, rate, nil
}

// ReportAddressCheckResults logs the results of an address check
func ReportAddressCheckResults(ctx *AppContext, numKeys, matched int, duration time.Duration, rate float64) {
	// Log total checks performed
	logger.LogHeaderStatus(ctx.LocalLog, constants.LogInfo,
		"Checked:    %s addresses in %.3f seconds",
		utils.FormatWithCommas(numKeys),
		duration.Seconds())

	// Log checking rate
	logger.LogStatus(ctx.LocalLog, constants.LogInfo,
		"Check Rate: %s addresses/second",
		utils.FormatWithCommas(int(rate)))

	// Log match statistics
	matchRate := float64(matched) / float64(numKeys) * 100
	logger.LogStatus(ctx.LocalLog, constants.LogInfo,
		"Matches:    %s (%.1f%% match rate)",
		utils.FormatWithCommas(matched),
		matchRate)

	// Verify match rate is close to 50%
	if matchRate < 45 || matchRate > 55 {
		logger.LogStatus(ctx.LocalLog, constants.LogWarn,
			"Unexpected match rate: %.1f%% (expected ~50%%)",
			matchRate)
	}

	// Log memory usage if in debug mode
	if constants.DebugMode {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		logger.LogStatus(ctx.LocalLog, constants.LogMem,
			"Memory usage: %.2f GB",
			float64(m.Alloc)/(1024*1024*1024))
	}
}
