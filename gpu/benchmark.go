package gpu

import (
	"Grendel/constants"
	"Grendel/generator"
	"Grendel/logger"
	"Grendel/utils"
	"fmt"
	"log"
	"os"
	"time"
)

type BenchResults struct {
	KeysGenerated int
	TimeElapsed   time.Duration
	KeysPerSecond float64
}

func RunBenchmark(count int, gen *CUDAGenerator) error {
	gpuLog := log.New(os.Stdout, "", 0)
	logger.LogHeaderStatus(gpuLog, constants.LogVideo,
		"CPU vs GPU Benchmark (%s addresses)",
		utils.FormatWithCommas(count))

	// CPU Benchmark
	cpuStart := time.Now()
	cpuGen, _ := NewCPUGenerator(&generator.Config{})
	_, _, _, err := cpuGen.Generate(count)
	if err != nil {
		return fmt.Errorf("CPU benchmark failed: %v", err)
	}
	cpuTime := time.Since(cpuStart)
	cpuRate := float64(count) / cpuTime.Seconds()

	// GPU Benchmark
	gpuStart := time.Now()
	buffer := make([]KeyAddressData, count)
	if err := gen.generateCombined(buffer, count); err != nil {
		return fmt.Errorf("GPU benchmark failed: %v", err)
	}
	gpuTime := time.Since(gpuStart)
	gpuRate := float64(count) / gpuTime.Seconds()

	cpuRateStr := utils.FormatWithCommas(int(cpuRate))
	gpuRateStr := utils.FormatWithCommas(int(gpuRate))

	logger.LogHeaderStatus(gpuLog, constants.LogVideo,
		"CPU: %v (%s keys/sec)",
		cpuTime, cpuRateStr)
	logger.LogStatus(gpuLog, constants.LogVideo,
		"GPU: %v (%s keys/sec)",
		gpuTime, gpuRateStr)
	logger.LogStatus(gpuLog, constants.LogVideo,
		"Speedup: %.2fx",
		gpuRate/cpuRate)

	logger.PrintSeparator(constants.LogVideo)
	return nil
}
