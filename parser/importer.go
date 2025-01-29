package parser

import (
	"Grendel/addresses"
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"encoding/hex" // Add this import
	"fmt"
	"os"
	"strings"

	"github.com/syndtr/goleveldb/leveldb"
)

func (p *Parser) ImportAddressesFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open addresses file: %v", err)
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf("failed to create gzip reader: %v", err)
	}
	defer gzReader.Close()

	scanner := bufio.NewScanner(gzReader)
	batch := new(leveldb.Batch)
	count := 0

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, ",")
		if len(parts) != 2 {
			continue
		}

		addr := parts[0]
		balanceStr := strings.TrimSpace(parts[1])

		// Balance should be exactly 16 hex chars (8 bytes)
		if len(balanceStr) != 16 {
			fmt.Printf("Skipping address %s - invalid balance hex length: %d\n", addr, len(balanceStr))
			continue
		}

		balance, err := hex.DecodeString(balanceStr)
		if err != nil {
			fmt.Printf("Failed to decode balance for address %s: %v\n", addr, err)
			continue
		}

		if len(balance) != 8 {
			fmt.Printf("Invalid balance length for address %s: got %d bytes\n", addr, len(balance))
			continue
		}

		// Store in database with original address
		batch.Put([]byte("addr:"+addr), balance)

		// Store in memory map with hash
		hash := fastHash(addr)
		p.addresses[hash] = binary.LittleEndian.Uint64(balance)

		// Categorize address
		addresses.CategorizeAddress(addr, p.Stats)

		count++

		if count == 1 {
			// Print first entry for debugging
			fmt.Printf("First entry:\nAddress: %s\nBalance (hex): %x\nBalance (len): %d\n",
				addr, balance, len(balance))
		}

		// Commit batch every 100,000 addresses
		if count%100000 == 0 {
			if err := p.DB.Write(batch, nil); err != nil {
				return fmt.Errorf("failed to write batch to db: %v", err)
			}
			batch.Reset()
		}
	}

	// Check for scanner errors
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading addresses: %v", err)
	}

	// Write final batch
	if err := p.DB.Write(batch, nil); err != nil {
		return fmt.Errorf("failed to write final batch: %v", err)
	}

	p.AddressCount = uint64(count)

	return nil
}
