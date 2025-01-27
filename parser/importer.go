package parser

import (
	"Grendel/addresses"
	"bufio"
	"compress/gzip"
	"os"
	"strings"

	"github.com/syndtr/goleveldb/leveldb"
)

func (p *Parser) ImportAddressesFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return err
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
		balance := []byte(parts[1])

		// Store in database
		batch.Put([]byte("addr:"+addr), balance)

		// Store in memory map
		p.addresses[addr] = string(balance)

		// Categorize address
		addresses.CategorizeAddress(addr, p.Stats)

		count++

		// Commit batch every 100,000 addresses
		if count%100000 == 0 {
			if err := p.DB.Write(batch, nil); err != nil {
				return err
			}
			batch.Reset()
		}
	}

	// Write final batch
	if err := p.DB.Write(batch, nil); err != nil {
		return err
	}

	p.AddressCount = uint64(count)
	return nil
}
