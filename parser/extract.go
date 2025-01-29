package parser

import (
	"Grendel/constants"
	"Grendel/utils"
	"bufio"
	"compress/gzip"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/syndtr/goleveldb/leveldb"
)

func ExtractAddresses() error {
	// Use proper database path
	dbPath := filepath.Join(utils.GetBaseDir(), constants.AddressDBPath)

	// Open the database
	db, err := leveldb.OpenFile(dbPath, nil)
	if err != nil {
		return fmt.Errorf("failed to open database: %v", err)
	}
	defer db.Close()

	// Ensure config directory exists
	if err := os.MkdirAll(".config", 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %v", err)
	}

	// Create output file with gzip compression
	outFile, err := os.Create(".config/addresses.txt.gz")
	if err != nil {
		return fmt.Errorf("failed to create output file: %v", err)
	}
	defer outFile.Close()

	// Create gzip writer
	gzWriter := gzip.NewWriter(outFile)
	defer gzWriter.Close()

	writer := bufio.NewWriter(gzWriter)

	// Iterate through database
	iter := db.NewIterator(nil, nil)
	defer iter.Release()

	count := 0
	skipped := 0
	for iter.Next() {
		key := iter.Key()
		if len(key) > 5 && string(key[:5]) == "addr:" {
			addr := string(key[5:])
			balance := iter.Value()

			// Skip invalid balances but count them
			if len(balance) != 8 {
				skipped++
				continue
			}

			// Write address and hex-encoded balance
			_, err := fmt.Fprintf(writer, "%s,%x\n", addr, balance)
			if err != nil {
				log.Printf("Error writing to file: %v", err)
				continue
			}
			count++
		}
	}

	// Flush the writer
	writer.Flush()
	gzWriter.Close()

	fmt.Printf("Exported %s addresses to addresses.txt.gz\n",
		utils.FormatWithCommas(count))
	if skipped > 0 {
		fmt.Printf("Skipped %d addresses with invalid balances\n", skipped)
	}
	return iter.Error()
}
