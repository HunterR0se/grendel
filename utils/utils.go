package utils

import (
	"Grendel/constants"
	"bufio"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/shirou/gopsutil/mem"
)

// CUDA checker
//
// CheckCUDA checks for the availability of CUDA on the system
func CheckCUDA() bool {
	cudaPaths := []string{
		"/opt/cuda/lib64/libcudart.so",
		"/opt/cuda/lib64/libcudart.so.12",
		"/usr/local/cuda/lib64/libcudart.so",
		"/usr/local/cuda/lib64/libcudart.so.12",
		"/usr/local/cuda-12.0/lib64/libcudart.so",
		"/usr/local/cuda-12.0/lib64/libcudart.so.12",
		"/usr/lib64/libcudart.so",
		"/usr/lib64/libcudart.so.12",
		"/usr/lib/x86_64-linux-gnu/libcudart.so",
		"/usr/lib/x86_64-linux-gnu/libcudart.so.12",
	}

	for _, path := range cudaPaths {
		if FileExists(path) {
			return true
		}
	}

	return false
}

func CalculateDBSize(dbPath string) float64 {
	files, err := os.ReadDir(dbPath)
	if err != nil {
		return 0
	}

	var totalSize int64
	for _, file := range files {
		info, err := file.Info()
		if err != nil {
			continue
		}
		// Count both .log and .ldb files
		if strings.HasSuffix(file.Name(), ".log") ||
			strings.HasSuffix(file.Name(), ".ldb") {
			totalSize += info.Size()
		}
	}
	return float64(totalSize) / (1024 * 1024 * 1024) // Convert to GB
}

// FileExists checks if a file exists at the given path
func FileExists(path string) bool {
	_, err := os.Stat(path)
	return !os.IsNotExist(err)
}

const (
	gb = 1 << 30 // 1GB
	mb = 1 << 20 // 1MB
)

type MemoryLimits struct {
	BlockCache     int
	WriteBuffer    int
	CompactionSize int
	BatchSize      int
	ChannelBuffer  int
	RNGPoolSize    int
	NumWorkers     int
	Available      float64
	Total          float64
}

func BoolToEnabledDisabled(b bool) string {
	if b {
		return "Enabled"
	}
	return "Disabled"
}

func GetBaseDir() string {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		log.Fatal("Could not find home directory:", err)
	}
	return homeDir
}
func GetAvailableDiskSpace() (float64, float64) {
	var stat syscall.Statfs_t
	err := syscall.Statfs(".", &stat)
	if err != nil {
		return 0, 0
	}
	available := stat.Bavail * uint64(stat.Bsize)
	total := stat.Blocks * uint64(stat.Bsize)
	return float64(available) / float64(1<<30), float64(total) / float64(1<<30)
}

// ObfuscationKey retrieves the obfuscation key from the Bitcoin debug.log file
func ObfuscationKey() ([]byte, error) {
	// Define the path to the Bitcoin data directory
	bitcoinDir := filepath.Join(os.Getenv("HOME"), ".bitcoin")
	debugLogPath := filepath.Join(bitcoinDir, "debug.log")

	// Open the debug.log file
	file, err := os.Open(debugLogPath)
	if err != nil {
		return nil, fmt.Errorf("error: %v", err)
	}
	defer file.Close()

	// Create a scanner to read the file line by line
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		// Check if the line contains the obfuscation key
		if strings.Contains(line, "Using obfuscation key for blocksdir") {
			// Extract the key from the line
			parts := strings.Split(line, "'")
			if len(parts) >= 2 {
				keyStr := parts[1]
				// Convert the hexadecimal string to a byte slice
				key, err := hex.DecodeString(keyStr)
				if err != nil {
					return nil, fmt.Errorf("failed to decode obfuscation key from debug.log: %v", err)
				}
				return key, nil
			}
		}
	}

	// If we reach here, we didn't find the obfuscation key
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading debug.log at %s: %v", debugLogPath, err)
	}

	return nil, fmt.Errorf("obfuscation key not found in debug.log at %s. If you are running a pruned node, the obfuscation key might not be available, and the application will continue without it.", debugLogPath)
}

// CalculateMemoryLimits calculates optimal memory limits
// and sizes based on available system memory.
//
// The function allocates memory as follows:
// - Block Cache: 60% of target memory (min 4GB, max 32GB)
// - Write Buffer: 25% of target memory (min 1GB, max 12GB)
// - Compaction Size: 15% of target memory (min 256MB, max 6GB)
// - Batch Size: 10% of max possible items (min 500k, max 10M)
// - Channel Buffer: Matches batch size
// - RNG Pool Size: Matches batch size
// - Workers: CPU cores * 4
//
// If system memory cannot be determined, falls back to conservative defaults.
//
// Returns a MemoryLimits struct containing all calculated values and limits.
func CalculateMemoryLimits() *MemoryLimits {
	v, err := mem.VirtualMemory()
	if err != nil {
		fmt.Printf("Cannot Start. Memory Not available %v", err)
		fmt.Println()
		os.Exit(1)
	}

	availableRAM := float64(v.Available)
	totalRAM := float64(v.Total)
	targetMemory := totalRAM * constants.MemoryTarget

	// maxItems := int(totalRAM / structSize)
	// batchSize := max(min(maxItems/10, constants.MaxBlockFiles), constants.ImportBatchSize)

	return &MemoryLimits{
		BlockCache:     int(targetMemory * constants.BlockCacheProportion),
		WriteBuffer:    int(targetMemory * constants.WriteBufferProportion),
		CompactionSize: int(targetMemory * constants.CompactionSizeProportion),
		Available:      availableRAM / gb,
		Total:          totalRAM / gb,
		NumWorkers:     constants.NumWorkers,
		BatchSize:      constants.ImportBatchSize,
		ChannelBuffer:  constants.ImportBatchSize * 2,
		RNGPoolSize:    constants.RNGPoolSize,
	}
}

// PeriodicGC starts a background goroutine that periodically runs garbage collection
// and logs memory statistics. It waits 5 seconds before starting the collection routine
// to allow initial program startup. Once started, it collects and logs memory stats at
// a configurable interval specified by constants.ImportLogInterval.
//
// The logged memory statistics include:
// - Alloc: Currently allocated heap memory in MiB
// - TotalAlloc: Total allocated heap memory over program lifetime in MiB
// - Sys: Total memory obtained from system in MiB
//
// The function runs indefinitely until program termination. Memory statistics are
// written to stdout using constants.LogInfo format.
func PeriodicGC() {
	var m runtime.MemStats

	// Get system memory info once
	v, _ := mem.VirtualMemory()
	totalRAM := float64(v.Total)

	// Create printMemStats function to avoid code duplication
	printMemStats := func() {
		runtime.GC()
		runtime.ReadMemStats(&m)

		allocGB := float64(m.Alloc) / float64(gb)
		sysGB := float64(m.Sys) / float64(gb)
		allocPercent := (allocGB / (totalRAM / float64(gb))) * 100
		sysPercent := (sysGB / (totalRAM / float64(gb))) * 100

		separator := strings.Repeat("─", constants.LineLength)
		fmt.Printf("[🗑 TRASH ] %s\n", separator)
		fmt.Printf("[🗑 TRASH ] Garbage Collection ~ %.1fGB (%.1f%%) Used | %.1fGB (%.1f%%) System\n",
			allocGB, allocPercent, sysGB, sysPercent)
		fmt.Printf("[🗑 TRASH ] %s\n", separator)
	}

	go func() {
		// Regular interval checks
		ticker := time.NewTicker(constants.ImportLogInterval * 100)
		defer ticker.Stop()

		for range ticker.C {
			printMemStats()
		}
	}()
}

// number formatter
func FormatNumber(n float64) string {
	switch {
	case n >= 1_000_000: // millions
		return fmt.Sprintf("%.1fM", n/1_000_000)
	case n >= 1_000: // thousands
		return fmt.Sprintf("%.1fk", n/1_000)
	}
	return strconv.FormatFloat(n, 'f', 0, 64)
}

func FormatWithCommas(n int) string {
	str := strconv.Itoa(n)
	for i := len(str) - 3; i > 0; i -= 3 {
		str = str[:i] + "," + str[i:]
	}
	return str
}

func FormatWithFixedCommas(n int) string {
	// Convert the number to a string with commas
	str := strconv.FormatInt(int64(n), 10)
	var buf strings.Builder

	// Calculate the length of the string with commas
	commaLen := len(str)
	// Calculate the total padding needed
	totalPadding := 12 - commaLen

	// If the number is negative, adjust padding
	if n < 0 {
		totalPadding--
	}

	// Write the padding to the buffer
	for i := 0; i < totalPadding; i++ {
		buf.WriteByte(' ')
	}

	// Write the number with commas to the buffer
	buf.WriteString(str)

	return buf.String()
}

// Add new function for formatted numbers with units
func FormatNumberWithUnit(n float64, unit string) string {
	str := fmt.Sprintf("%.1f", n)
	parts := strings.Split(str, ".")

	intPart := parts[0]
	for i := len(intPart) - 3; i > 0; i -= 3 {
		intPart = intPart[:i] + "," + intPart[i:]
	}

	if len(parts) > 1 {
		return fmt.Sprintf("%9s.%s%s", intPart, parts[1], unit)
	}
	return fmt.Sprintf("%9s%s", intPart, unit)
}

type foundData struct {
	Address string `json:"address"`
	Balance int64  `json:"balance"`
	Time    string `json:"time"`
}

func WriteFound(addr string, balance int64) error {
	found := foundData{
		Address: addr,
		Balance: balance,
		Time:    time.Now().Format(time.RFC3339),
	}

	data, err := json.MarshalIndent(found, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal error: %w", err)
	}

	return os.WriteFile("found.json", data, 0644)
}

// GetFreeDiskSpace returns available disk space in GB
func GetFreeDiskSpace() uint64 {
	var stat syscall.Statfs_t

	// Get statistics from current directory
	err := syscall.Statfs(".", &stat)
	if err != nil {
		return 0
	}

	// Calculate available space in GB
	// Available blocks * block size
	available := stat.Bavail * uint64(stat.Bsize)
	return available >> 30 // Convert to GB (divide by 2^30)
}

// Helper function to split messages into multiple lines
func SplitMessage(message string, maxLen int, prefix string) []string {
	var lines []string

	// First line uses original prefix
	lines = append(lines, message[:min(len(message), constants.LineLength)])

	// If there's more content, add continuation lines
	if len(message) > constants.LineLength {
		remaining := message[constants.LineLength:]
		for len(remaining) > 0 {
			lineLen := min(len(remaining), maxLen)
			lines = append(lines, fmt.Sprintf("%s ... %s", prefix, remaining[:lineLen]))
			if len(remaining) > lineLen {
				remaining = remaining[lineLen:]
			} else {
				remaining = ""
			}
		}
	}

	return lines
}

// Helper function to find minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
