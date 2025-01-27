package loader

import (
	"Grendel/constants"
	"Grendel/logger"
	"Grendel/parser"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/syndtr/goleveldb/leveldb"
)

const (
	logInterval = time.Second
)

func initDB(p *parser.Parser) error {
	if err := os.MkdirAll(p.DBPath, 0755); err != nil {
		logger.LogError(p.Logger, constants.LogError, err, "Failed to create database directory")
		return fmt.Errorf("failed to create database directory: %v", err)
	}
	var err error
	p.DB, err = leveldb.OpenFile(p.DBPath, nil)
	if err != nil {
		logger.LogError(p.Logger, constants.LogError, err, "Failed to initialize database")
		return fmt.Errorf("failed to initialize database: %v", err)
	}
	return nil
}

func writeBatch(p *parser.Parser, batch *leveldb.Batch) error {
	if err := p.DB.Write(batch, nil); err != nil {
		logger.LogError(p.Logger, constants.LogError, err, "Error writing batch")
		return fmt.Errorf("error writing batch: %v", err)
	}
	return nil
}

func cleanupDB(p *parser.Parser) error {
	if p.DB != nil {
		p.DB.Close()
	}
	if err := os.RemoveAll(p.DBPath); err != nil && !os.IsNotExist(err) {
		logger.LogError(p.Logger, constants.LogError, err, "Failed to cleanup database")
		return fmt.Errorf("failed to cleanup database: %v", err)
	}
	return nil
}

func parseAddressLine(line string) (string, int64, error) {
	if i := strings.IndexByte(line, ','); i >= 0 {
		address := strings.TrimSpace(line[:i])
		balance, err := strconv.ParseInt(strings.TrimSpace(line[i+1:]), 10, 64)
		if err != nil {
			return "", 0, err
		}
		return address, balance, nil
	}
	return "", 0, fmt.Errorf("invalid line format")
}
