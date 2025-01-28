package logger

import (
	"Grendel/constants"
	"Grendel/utils"
	"fmt"
	"log"
	"strings"
	"time"
)

var (
	lastLogType   string
	lineCounter   int64
	currentHeader string
)

// Declare global variables to store previous values
var (
	prevLegacy float64
	prevSegwit float64
	prevNative float64
)

func FirstLoad() {
	// Initialize prevLegacy, prevSegwit, and prevNative with values from GlobalStats
	prevLegacy = float64(constants.GlobalStats.LegacyCount)
	prevSegwit = float64(constants.GlobalStats.SegwitCount)
	prevNative = float64(constants.GlobalStats.NativeCount)
}

const (
	HeaderBlock = "BLOCKS  TIME | HEIGHT |  TOTALS  |  AVERAGE RATE | SYSRAM | DISK"
)

var (
	HeaderGenerate = "GENERATED |  RATE/s | LEGACY  |  SEGWIT |  NATIVE |  RAM  | FIND"
)

// Logger is a custom logger type
type Logger struct {
	*log.Logger
}

// NewLogger creates a new Logger instance
func NewLogger() *Logger {
	return &Logger{
		Logger: log.New(log.Writer(), "", log.LstdFlags),
	}
}

func PrintSeparator(logType string) {
	// Create separator line with dynamic length
	separator := strings.Repeat("─", constants.LineLength)
	fmt.Printf("%s%s\n", logType, separator)
}

func Banner() {
	fmt.Printf(`

   ________                          .___     .__           🛠️ Build 104 🛠️
  /  _____/______   ____   ____    __| _/____ |  |  🕹️ CUDA Acceleration 🕹️
 /   \  __\_  __ \_/ __ \ /    \  / __ |/ __ \|  |  Updated Jan 27, 2025 ⏱️
 \    \_\  \  | \/\  ___/|   |  \/ /_/ \  ___/|  |__
  \________/__|    \____/\___|__/\_____|\____/_____/   🌹 by Hunter Rose 🌹

`)
}

// LogError standardizes error logging with line wrapping
func LogError(logger *log.Logger, prefix string, err error, context string) {
	var message string
	if context != "" {
		message = fmt.Sprintf("%s %s: %v", prefix, context, err)
	} else {
		message = fmt.Sprintf("%s Error: %v", prefix, err)
	}

	// Calculate max length for content after prefix
	maxLen := constants.LineLength - len(prefix) - 1 // -1 for space after prefix

	// If message is longer than max length, split it
	if len(message) > constants.LineLength {
		lines := utils.SplitMessage(message, maxLen, prefix)
		for _, line := range lines {
			logger.Print(line)
		}
	} else {
		logger.Print(message)
	}
}

// LogDebug standardizes debug logging
func LogDebug(logger *log.Logger, prefix string, format string, args ...interface{}) {
	if constants.DebugMode && format != "" {
		// Use fmt.Print instead of logger.Printf to avoid timestamp
		fmt.Printf("%s%s\n", prefix, fmt.Sprintf(format, args...))
	}
}

// LogStatus standardizes status/info logging with line wrapping
func LogStatus(logger *log.Logger, prefix string, message string, args ...interface{}) {
	if logger == nil {
		logger = log.Default()
	}

	// Format the message with args
	msg := fmt.Sprintf(message, args...)
	fullMessage := fmt.Sprintf("%s%s", prefix, msg)

	// Calculate max length for content after prefix
	maxLen := constants.LineLength - len(prefix) - 1 // -1 for space after prefix

	// If message is longer than max length, split it
	if len(fullMessage) > constants.LineLength {
		lines := utils.SplitMessage(fullMessage, maxLen, prefix)
		for _, line := range lines {
			logger.Print(line)
		}
	} else {
		logger.Print(fullMessage)
	}
}

// LogStatus standardizes status/info logging
func LogHeaderStatus(logger *log.Logger,
	prefix string,
	message string,
	args ...interface{}) {

	if logger == nil {
		logger = log.Default()
	}
	// Trim message and combine prefix to stay within line length
	msg := fmt.Sprintf(message, args...)
	maxLen := constants.LineLength - len(prefix) - 1 // Account for prefix and space
	if len(msg) > maxLen {
		msg = msg[:maxLen-1]
	}
	PrintSeparator(prefix)
	logger.Printf("%s%s", prefix, msg)
}

func logWithTypeChange(logger *log.Logger, logType string, message string) {
	// Increment counter first
	lineCounter++

	// Check for header printing before resetting counter
	if lastLogType != logType || lineCounter%42 == 0 {
		var header string
		switch logType {
		case constants.LogDB, constants.LogStats:
			header = HeaderGenerate
		case constants.LogParser:
			header = HeaderBlock
		}

		// Only print header if it's different from current header
		if header != "" && header != currentHeader {
			PrintSeparator(constants.LogHeader)
			logger.Printf("%s %s", constants.LogHeader, header)
			PrintSeparator(constants.LogHeader)
			currentHeader = header
		}

		// Reset counter only on type change
		if lastLogType != logType {
			lineCounter = 0
		}
	}

	logger.Print(message)
	lastLogType = logType

	// Reset counter after 42 lines
	if lineCounter >= 42 {
		lineCounter = 0
	}
}

func LogGeneratorStats(
	logger *log.Logger,
	generated float64,
	rate float64,
	elapsed time.Duration,
	memGB float64,
	found uint64,
	legacy float64,
	segwit float64,
	native float64,
) {
	// Format rate to be more reasonable (k/s instead of raw count)
	rateInK := rate / 1000 // Convert to thousands per second

	// Calculate newly generated addresses since last log
	newLegacy := legacy - prevLegacy
	newSegwit := segwit - prevSegwit
	newNative := native - prevNative

	// Update global variables with current values for next iteration
	prevLegacy = legacy
	prevSegwit = segwit
	prevNative = native

	droppedCount := int(constants.TotalDroppedAddresses.Load())
	finalCount := 0

	// Before displaying stats, check if we have any FOUND addresses
	if found > 0 {
		finalCount = int(found)
		HeaderGenerate = "GENERATED |  RATE/s | LEGACY  |  SEGWIT |  NATIVE |  RAM  | FIND"
	} else {
		finalCount = droppedCount
		HeaderGenerate = "GENERATED |  RATE/s | LEGACY  |  SEGWIT |  NATIVE |  RAM  | DROP"
	}

	message := fmt.Sprintf("[%s] %9sM | %6.1fk | %6.3fM | %6.3fM | %6.3fM | %4.1fG | %d",
		time.Now().Format("15:04:05"),
		utils.FormatWithCommas(int(generated)),
		rateInK,
		newLegacy/1_000_000,
		newSegwit/1_000_000,
		newNative/1_000_000,
		memGB,
		finalCount)

	// Reset TotalDroppedAddresses back to 0 after logging
	constants.TotalDroppedAddresses.Store(0)

	logWithTypeChange(logger, constants.LogDB, message)
}

// block progress logger
func LogBlockProgress(
	logger *log.Logger,
	blocks int,
	startTime time.Time,
	lastLogTime time.Time,
	height int64,
	totalAddrs int,
	memGB float64,
	diskFreeGB uint64,
) {
	now := time.Now()
	timeSinceLastLog := now.Sub(lastLogTime)
	elapsedSinceStart := now.Sub(startTime)

	// Format blocks - only show as K if actually over 1000
	var blocksFormatted string
	if blocks >= 1000 {
		blocksFormatted = fmt.Sprintf("%.1fK", float64(blocks)/1e3)
	} else {
		blocksFormatted = fmt.Sprintf("%d", blocks)
	}

	// Format total addresses in millions
	totalAddrsFormatted := fmt.Sprintf("%.3fM", float64(totalAddrs)/1e6)

	// Calculate instantaneous rate
	var ratePerSec float64
	if timeSinceLastLog.Seconds() > 0 {
		ratePerSec = float64(totalAddrs) / timeSinceLastLog.Seconds()
	}
	rateFormatted := utils.FormatWithCommas(int(ratePerSec))

	// Calculate average rate since start
	var averageRatePerSec float64
	if elapsedSinceStart.Seconds() > 0 {
		averageRatePerSec = float64(totalAddrs) / elapsedSinceStart.Seconds()
	}
	averageRateFormatted := utils.FormatWithCommas(int(averageRatePerSec))
	_ = averageRateFormatted

	// Format elapsed time since start of block processing
	hours := int(elapsedSinceStart.Hours())
	minutes := int(elapsedSinceStart.Minutes()) % 60
	timeStr := fmt.Sprintf("%02d:%02d", hours, minutes)

	message := fmt.Sprintf("[%02d:%02d:%02d] %7s %5s | %6d | %8s | %11s/s | %5.1fG | %3dG",
		now.Hour(),
		now.Minute(),
		now.Second(),
		blocksFormatted,
		timeStr,
		height,
		totalAddrsFormatted,
		rateFormatted,
		// averageRateFormatted,
		memGB,
		int(diskFreeGB))

	// Use logWithTypeChange to handle header logic consistently
	logWithTypeChange(logger, constants.LogParser, message)
}
