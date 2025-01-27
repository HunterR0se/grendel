package parser

import (
	"bufio"
	"compress/gzip"
	"fmt"
	"log"
	"os"

	"github.com/syndtr/goleveldb/leveldb"
)

func ExtractAddresses() error {
	// Path to your LevelDB database
	dbPath := ".addresses.db" // adjust this path

	// Open the database
	db, err := leveldb.OpenFile(dbPath, nil)
	if err != nil {
		log.Fatalf("Failed to open database: %v", err)
	}
	defer db.Close()

	// Create output file with gzip compression
	outFile, err := os.Create("config/addresses.txt.gz")
	if err != nil {
		log.Fatalf("Failed to create output file: %v", err)
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
	for iter.Next() {
		key := iter.Key()
		if len(key) > 5 && string(key[:5]) == "addr:" {
			addr := string(key[5:])
			balance := iter.Value()

			// Write address and balance to file
			_, err := fmt.Fprintf(writer, "%s,%s\n", addr, balance)
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

	fmt.Printf("Exported %d addresses to addresses.txt.gz\n", count)
	return err
}
