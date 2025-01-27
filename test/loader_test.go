package test

import (
	"Grendel/constants"
	"Grendel/loader"
	"Grendel/logger"
	"encoding/hex"
	"log"
	"os"
	"testing"

	"github.com/btcsuite/btcd/chaincfg"
)

var testLogger *log.Logger

func init() {
	testLogger = log.New(os.Stdout, "", log.LstdFlags)
	constants.DebugMode = true // Enable debug mode for detailed logging during tests
}

// TestExtractInputAddresses tests the ExtractInputAddresses function
func TestExtractInputAddresses(t *testing.T) {
	// Test case 1: Legacy address (P2PKH)
	p2pkhScript, _ := hex.DecodeString("76a91462e907b15cbf27d5425399ebf6f0fb50ebb88f1888ac")
	testExtractInputAddresses(t, p2pkhScript, "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "Legacy address should be extracted correctly")

	// Test case 2: SegWit address (P2WPKH)
	p2wpkhScript, _ := hex.DecodeString("0014751e76e8199196d454941c45d1b3a323f1433bd6")
	testExtractInputAddresses(t, p2wpkhScript, "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4", "SegWit address should be extracted correctly")

	// Test case 3: Taproot address
	taprootScript, _ := hex.DecodeString("51201337e8bb305436072ae10e8bd096b64d5f0e9c67448a7a6ed9321809c60f7c9a")
	testExtractInputAddresses(t, taprootScript, "bc1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr", "Taproot address should be extracted correctly")
}

// Helper function to test ExtractInputAddresses
func testExtractInputAddresses(t *testing.T, script []byte, expectedAddress string, msg string) {
	addresses, err := loader.ExtractInputAddresses(script, &chaincfg.MainNetParams)
	if err != nil {
		t.Errorf("ExtractInputAddresses failed: %v", err)
		return
	}

	if len(addresses) == 0 {
		t.Errorf("No addresses extracted: %s", msg)
		return
	}

	found := false
	for _, addr := range addresses {
		if addr.EncodeAddress() == expectedAddress {
			found = true
			logger.LogDebug(testLogger, constants.LogDebug, "Extracted address: %s", addr.EncodeAddress())
			break
		}
	}

	if !found {
		t.Errorf("Expected address %s not found: %s", expectedAddress, msg)
	}
}
