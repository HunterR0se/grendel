package loader

import (
	"Grendel/addresses"
	"Grendel/constants"
	"Grendel/logger"
	"Grendel/utils"
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/btcsuite/btcd/chaincfg"
	"github.com/btcsuite/btcd/txscript"
	"github.com/btcsuite/btcd/wire"
	"github.com/shirou/gopsutil/disk"
	"github.com/syndtr/goleveldb/leveldb"
)

var startTime time.Time

func (bl *BlockLoader) FindBlockFiles() ([]string, error) {
	blocksDir := filepath.Join(bl.parser.BitcoinDir, "blocks")

	if _, err := os.Stat(blocksDir); err != nil {
		return nil, fmt.Errorf("blocks directory not found: %v", err)
	}

	entries, err := os.ReadDir(blocksDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read blocks directory: %v", err)
	}

	validFiles := make([]string, 0, len(entries)/2)

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		if !strings.HasPrefix(name, "blk") ||
			len(name) < 7 ||
			name == "xor.dat" ||
			strings.HasPrefix(name, "rev") {
			continue
		}

		numStr := name[3 : len(name)-4]
		if _, err := strconv.Atoi(numStr); err != nil {
			continue
		}

		fullPath := filepath.Join(blocksDir, name)
		if !bl.parser.IsBlockProcessed(fullPath) {
			validFiles = append(validFiles, fullPath)
		}
	}

	if len(validFiles) > 0 {
		if constants.DebugMode {
			logger.LogHeaderStatus(bl.parser.Logger, constants.LogImport,
				"Found %d new blocks to process", len(validFiles))
		}
	}

	return validFiles, nil
}

func (bl *BlockLoader) processBlockFile(path string) (map[string]int64, int, error) {
	logger.LogDebug(bl.parser.Logger, constants.LogImport,
		"Processing block file: %s", path)

	file, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to open block file: %v", err)
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return nil, 0, fmt.Errorf("failed to get file info: %v", err)
	}
	fileSize := fileInfo.Size()

	reader := bufio.NewReaderSize(file, 1024*1024)
	localAddressMap := make(map[string]int64, 10000)

	blocksInFile := 0
	bytesRead := int64(0)
	totalTx := 0
	totalAddresses := 0

	blockData := make([]byte, 4*1024*1024)
	magicBuf := make([]byte, 4)
	sizeBuf := make([]byte, 4)

	basename := filepath.Base(path)
	isRev := strings.HasPrefix(basename, "rev")
	isBlk := strings.HasPrefix(basename, "blk")

	if isRev {
		return localAddressMap, blocksInFile, nil
	}

	for bytesRead < fileSize {
		if _, err := io.ReadFull(reader, magicBuf); err != nil {
			if err == io.EOF && bytesRead == fileSize {
				break
			}
			return nil, blocksInFile, fmt.Errorf("failed to read magic bytes %d: %v", bytesRead, err)
		}
		bytesRead += 4

		if isBlk && constants.ObfuscationKey != nil {
			for i := 0; i < 4; i++ {
				magicBuf[i] ^= constants.ObfuscationKey[i]
			}

			if magicBuf[0] != 0xF9 || magicBuf[1] != 0xBE ||
				magicBuf[2] != 0xB4 || magicBuf[3] != 0xD9 {
				bytesRead++
				continue
			}
		}

		if _, err := io.ReadFull(reader, sizeBuf); err != nil {
			return nil, blocksInFile, fmt.Errorf("failed to read block size %d: %v", bytesRead, err)
		}
		bytesRead += 4

		if isBlk && constants.ObfuscationKey != nil {
			for i := 0; i < 4; i++ {
				sizeBuf[i] ^= constants.ObfuscationKey[i+4]
			}
		}

		blockSize := binary.LittleEndian.Uint32(sizeBuf)

		if uint32(len(blockData)) < blockSize {
			blockData = make([]byte, blockSize)
		}
		blockData = blockData[:blockSize]

		if _, err := io.ReadFull(reader, blockData); err != nil {
			return nil, blocksInFile, err
		}
		bytesRead += int64(blockSize)

		if isBlk && constants.ObfuscationKey != nil {
			for i := 0; i < int(blockSize); i++ {
				blockData[i] ^= constants.ObfuscationKey[i&7]
			}
		}

		block := wire.MsgBlock{}
		if err := block.Deserialize(bytes.NewBuffer(blockData)); err != nil {
			logger.LogError(bl.parser.Logger, constants.LogError, err,
				"Failed to deserialize block")
			continue
		}

		blocksInFile++
		txCount := len(block.Transactions)
		totalTx += txCount

		for _, tx := range block.Transactions {
			if !IsCoinbaseTx(tx) {
				for _, txIn := range tx.TxIn {
					addrs, err := ExtractInputAddresses(txIn.SignatureScript, &chaincfg.MainNetParams)
					if err == nil {
						for _, addr := range addrs {
							addrStr := addr.String()
							// Track input addresses if TrackAllAddresses is enabled
							if constants.TrackAllAddresses {
								if _, exists := localAddressMap[addrStr]; !exists {
									localAddressMap[addrStr] = 0
									totalAddresses++
									if constants.DebugMode {
										logger.LogDebug(bl.parser.Logger,
											constants.LogDebug,
											"Found input address: %s", addrStr)
									}
								}
							}
						}
					}
				}
			}

			for _, txOut := range tx.TxOut {
				_, addrs, _, err := txscript.ExtractPkScriptAddrs(
					txOut.PkScript,
					&chaincfg.MainNetParams)

				if err != nil {
					addrs, err = ParseComplexScript(txOut.PkScript)
				}

				if err == nil && len(addrs) > 0 {
					for _, addr := range addrs {
						addrStr := addr.String()
						// Always track output addresses with balances
						localAddressMap[addrStr] += txOut.Value
						totalAddresses++

						// If TrackAllAddresses is false, only log addresses with positive balances
						if constants.DebugMode && (constants.TrackAllAddresses || txOut.Value > 0) {
							logger.LogDebug(bl.parser.Logger, constants.LogDebug,
								"Found output address: %s with value %d",
								addrStr, txOut.Value)
						}
					}
				}
			}
		}

		if blocksInFile%1000 == 0 {
			logger.LogStatus(bl.parser.Logger, constants.LogImport,
				"Block Progress: %d blocks, %d tx, %d addresses found",
				blocksInFile, totalTx, len(localAddressMap))
		}
	}

	return localAddressMap, blocksInFile, nil
}

func (bl *BlockLoader) GetLastBlockHeight() (int64, error) {
	key := []byte("last_block_height")
	value, err := bl.parser.DB.Get(key, nil)
	if err == leveldb.ErrNotFound {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	return int64(binary.LittleEndian.Uint64(value)), nil
}

func (bl *BlockLoader) UpdateBlockProgress(height int64) error {
	bl.parser.LastBlockHeight = height
	key := []byte("last_block_height")
	value := make([]byte, 8)
	binary.LittleEndian.PutUint64(value, uint64(height))
	return bl.parser.DB.Put(key, value, nil)
}

func (bl *BlockLoader) LogAddressSummary(title string, forceShow bool) {
	if !forceShow && !constants.DebugMode {
		return
	}

	constants.GlobalStats.RLock()
	defer constants.GlobalStats.RUnlock()

	logger.LogHeaderStatus(bl.parser.Logger, constants.LogHeader, "%s", title)
	logger.PrintSeparator(constants.LogHeader)
	logger.LogStatus(bl.parser.Logger, constants.LogStats,
		"Total Addresses: %11s",
		utils.FormatWithCommas(int(constants.GlobalStats.TotalCount)))
	logger.LogStatus(bl.parser.Logger, constants.LogStats,
		"     Legacy (1): %11s",
		utils.FormatWithCommas(int(constants.GlobalStats.LegacyCount)))
	logger.LogStatus(bl.parser.Logger, constants.LogStats,
		"     SegWit (3): %11s",
		utils.FormatWithCommas(int(constants.GlobalStats.SegwitCount)))
	logger.LogStatus(bl.parser.Logger, constants.LogStats,
		"   Native (bc1): %11s",
		utils.FormatWithCommas(int(constants.GlobalStats.NativeCount)))
	if constants.GlobalStats.WScriptCount > 0 {
		logger.LogStatus(bl.parser.Logger, constants.LogStats,
			"  Script (bc1w): %11s",
			utils.FormatWithCommas(int(constants.GlobalStats.WScriptCount)))
	}
	logger.PrintSeparator(constants.LogStats)
}

func (bl *BlockLoader) loadAddressesFromDB() error {
	if constants.DebugMode {
		fmt.Printf("Loading addresses from %s\n", bl.parser.DBPath)
	}

	// Now try to find addresses specifically
	count := uint64(0)
	iter := bl.parser.DB.NewIterator(nil, nil)
	defer iter.Release()

	for iter.Next() {
		key := iter.Key()
		if len(key) > 5 && string(key[:5]) == "addr:" {
			addr := string(key[5:])
			value := iter.Value()
			balance := int64(binary.LittleEndian.Uint64(value))
			if constants.DebugMode {
				fmt.Printf("Found address: %s with balance: %d\n", addr, balance)
			}
			addressMap[addr] = balance
			addresses.CategorizeAddress(addr, bl.parser.Stats)
			count++
		}
	}

	if err := iter.Error(); err != nil {
		return fmt.Errorf("error iterating database: %v", err)
	}

	// Update global stats after loading all addresses
	constants.GlobalStats.Lock()
	constants.GlobalStats.LegacyCount = int64(bl.parser.Stats.LegacyCount)
	constants.GlobalStats.SegwitCount = int64(bl.parser.Stats.SegwitCount)
	constants.GlobalStats.NativeCount = int64(bl.parser.Stats.NativeCount)
	constants.GlobalStats.TotalCount = constants.GlobalStats.LegacyCount +
		constants.GlobalStats.SegwitCount +
		constants.GlobalStats.NativeCount
	constants.GlobalStats.Unlock()

	if constants.DebugMode {
		fmt.Printf("Successfully loaded %d addresses from database\n", count)
	}
	bl.parser.AddressesFound = int64(count)
	return nil
}

func (bl *BlockLoader) LoadNewBlocks() error {
	bl.mutex.Lock()
	defer bl.mutex.Unlock()

	// Load BlocksProcessed from DB
	if err := bl.parser.LoadBlocksProcessed(); err != nil {
		return fmt.Errorf("failed to load blocks processed: %v", err)
	}

	blockFiles, err := bl.FindBlockFiles()
	if err != nil {
		return fmt.Errorf("failed to find block files: %v", err)
	}

	if startTime.IsZero() {
		startTime = time.Now() // Set the global startTime here if it hasn't been set yet
	}

	// Process ALL blocks if no addresses are found
	if bl.parser.AddressesFound == 0 {
		if constants.DebugMode {
			fmt.Printf("No addresses found in DB, reprocessing all blocks\n")
		}
		if err := bl.ProcessNewBlocks(blockFiles,
			bl.parser.BlocksProcessed,
			bl.parser.AddressesFound); err != nil {
			logger.LogError(bl.parser.Logger, constants.LogError, err,
				"Initial block processing failed but continuing")
			// Note: We're not returning the error here
		}
	} else {
		// Normal processing of new blocks
		var newFiles []string
		for _, file := range blockFiles {
			if !bl.parser.IsBlockProcessed(file) {
				newFiles = append(newFiles, file)
			}
		}

		if len(newFiles) > 0 {
			if err := bl.ProcessNewBlocks(newFiles,
				bl.parser.BlocksProcessed,
				bl.parser.AddressesFound); err != nil {
				logger.LogError(bl.parser.Logger, constants.LogError, err,
					"Block processing failed but continuing")
				// Note: We're not returning the error here
			}
		}
	}

	// Log success and continue even if some blocks failed
	if (int(bl.parser.AddressesFound)) > 0 {
		logger.LogHeaderStatus(bl.parser.Logger, constants.LogImport,
			"Found %s addresses (from blocks)",
			utils.FormatWithCommas(int(bl.parser.AddressesFound)))
	}

	// Signal that we're ready to move to address generation
	return nil
}

func (bl *BlockLoader) ProcessNewBlocks(blockFiles []string,
	initialBlocksProcessed,
	initialAddressesFound int64) error {

	lastLog := startTime // Use the global startTime
	batch := new(leveldb.Batch)
	addressesInBatch := 0

	// Track the highest block we've seen
	var highestBlock int64

	for i, blockFile := range blockFiles {
		if bl.parser.IsBlockProcessed(blockFile) {
			continue
		}

		// Skip non-block files
		filename := filepath.Base(blockFile)
		if !strings.HasPrefix(filename, "blk") ||
			strings.HasPrefix(filename, "rev") ||
			filename == "xor.dat" {
			logger.LogDebug(bl.parser.Logger, constants.LogDebug,
				"Skipping non-block file: %s", filename)
			continue
		}

		// Add progress logging every N blocks
		if i > 0 && i%10 == 0 {
			logger.LogDebug(bl.parser.Logger, constants.LogImport,
				"Processed %d/%d blocks (%d%%) - %s addresses",
				i, len(blockFiles),
				(i*100)/len(blockFiles),
				utils.FormatWithCommas(int(bl.parser.AddressesFound-initialAddressesFound)))
		}

		// Extract height from filename (only for blk*.dat files)
		height, err := strconv.ParseInt(strings.TrimPrefix(
			strings.TrimSuffix(filename, ".dat"),
			"blk"), 10, 64)
		if err != nil {
			logger.LogDebug(bl.parser.Logger, constants.LogDebug,
				"Skipping file with invalid block height: %s", filename)
			continue
		}

		if height > highestBlock {
			highestBlock = height
		}

		// Process single block file
		if err := bl.processSingleBlock(blockFile, batch, i, &lastLog,
			initialBlocksProcessed, initialAddressesFound); err != nil {
			return err
		}
	}

	// Write any remaining addresses in the final batch
	if addressesInBatch >= constants.ImportBatchSize {
		if err := bl.parser.DB.Write(batch, nil); err != nil {
			return fmt.Errorf("failed to write batch: %v", err)
		}
		batch.Reset()
		addressesInBatch = 0
	}

	// Update the last processed height
	if highestBlock > 0 {
		if err := bl.UpdateBlockProgress(highestBlock); err != nil {
			logger.LogError(bl.parser.Logger, constants.LogError, err,
				"Failed to update block progress")
		}
	}

	return nil
}

func (bl *BlockLoader) processSingleBlock(blockFile string,
	batch *leveldb.Batch,
	index int,
	lastLog *time.Time,
	initialBlocksProcessed,
	initialAddressesFound int64) error {

	// Add debug logging
	logger.LogDebug(bl.parser.Logger,
		constants.LogDebug,
		"Processing: %s", blockFile)

	logger.LogDebug(bl.parser.Logger, constants.LogDebug,
		"Index: %d Start %s Addresses %v", index, startTime, initialAddressesFound)

	// Check if we should process this block
	if !bl.shouldRetryBlock(blockFile) {
		if constants.DebugMode {
			logger.LogStatus(bl.parser.Logger, constants.LogDebug,
				"Skipping block %s - waiting for retry timeout", blockFile)
		}
		return nil
	}

	// Extract block height from filename (blk00000.dat -> 0)
	height, err := strconv.ParseInt(strings.TrimPrefix(
		strings.TrimSuffix(filepath.Base(blockFile), ".dat"),
		"blk"), 10, 64)
	if err != nil {
		logger.LogError(bl.parser.Logger, constants.LogError, err,
			fmt.Sprintf("Failed to parse block height from %s", blockFile))
		return nil
	}

	// Process block
	var localAddressMap map[string]int64
	var blockCount int
	localAddressMap, blockCount, err = bl.processBlockFile(blockFile)
	if err != nil {
		if strings.Contains(err.Error(), "failed to read magic bytes") {
			bl.markBlockForRetry(blockFile)
			return nil
		}
		logger.LogDebug(bl.parser.Logger, constants.LogError,
			"Failed %s: %s", blockFile, err.Error())
		return nil
	}

	// Add debug logging for address count
	if len(localAddressMap) > 0 {
		if constants.DebugMode {
			fmt.Printf("Found %d addresses in block %s\n",
				len(localAddressMap), blockFile)
		}
		// Print first few addresses as sample
		count := 0
		for addr := range localAddressMap {
			if count < 5 {
				if constants.DebugMode {
					fmt.Printf("Sample address: %s\n", addr)
				}
				count++
			} else {
				break
			}
		}
	}

	// Process addresses immediately
	addressesInBatch := 0
	for addr, balance := range localAddressMap {
		key := append([]byte("addr:"), addr...)
		existingBalance, err := bl.parser.DB.Get(key, nil)
		if err == leveldb.ErrNotFound {
			value := make([]byte, 8)
			binary.LittleEndian.PutUint64(value, uint64(balance))
			batch.Put(key, value)
		} else if err == nil {
			currentBalance := int64(binary.LittleEndian.Uint64(existingBalance))
			newBalance := currentBalance + balance
			value := make([]byte, 8)
			binary.LittleEndian.PutUint64(value, uint64(newBalance))
			batch.Put(key, value)
		} else {
			logger.LogError(bl.parser.Logger, constants.LogError, err, "Failed to get existing balance for address")
			continue
		}
		addresses.CategorizeAddress(addr, bl.parser.Stats)
		bl.parser.AddressesFound++
		addressesInBatch++
	}

	// Write any remaining addresses in the batch
	if addressesInBatch > 0 {
		if constants.DebugMode {
			fmt.Printf("Writing final batch of %d addresses from block %s\n",
				addressesInBatch, blockFile)
		}
		if err := bl.parser.DB.Write(batch, nil); err != nil {
			return fmt.Errorf("failed to write final batch: %v", err)
		}
		batch.Reset()
	}

	// Update progress
	bl.parser.BlocksProcessed += int64(blockCount)
	bl.parser.LastBlockHeight = bl.parser.BlocksProcessed

	// Log progress every ~60 seconds (keeping the rate limiting)
	now := time.Now()
	if time.Since(*lastLog) >= constants.ImportLogInterval {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		memGB := float64(m.Sys) / (1024 * 1024 * 1024)
		memGB = math.Round(memGB*100) / 100

		// Get disk usage
		diskInfo, _ := disk.Usage(bl.parser.BitcoinDir)
		diskFreeGB := uint64(0)
		if diskInfo != nil {
			diskFreeGB = diskInfo.Free / (1024 * 1024 * 1024)
		}

		logger.LogBlockProgress(
			bl.parser.Logger,
			int(bl.parser.BlocksProcessed-initialBlocksProcessed),
			startTime, // Use the global startTime here
			*lastLog,
			height,
			int(bl.parser.AddressesFound),
			memGB,
			diskFreeGB,
		)

		*lastLog = now
	}

	// Mark block as processed
	bl.parser.MarkBlockProcessed(blockFile)

	return nil
}
