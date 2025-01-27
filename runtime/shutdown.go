package runtime

import (
	"Grendel/constants"
	"Grendel/logger"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
)

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
func SetupGracefulShutdown(localLog *log.Logger,
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
		CloseOnce.Do(func() {
			close(shutdownChan)
			close(doneChan) // Close doneChan to make goroutines exit immediately
		})
	}()

	return shutdownChan
}
