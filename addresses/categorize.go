package addresses

import (
	"Grendel/constants"
	"Grendel/logger"
	"log"
	"os"
	"sync"

	"github.com/btcsuite/btcd/chaincfg"
	"github.com/btcsuite/btcutil"
)

// Move mutex to package level
var categoryMutex sync.RWMutex

func CategorizeAddress(addr string, category *constants.AddressCategory) {
	decodedAddr, err := btcutil.DecodeAddress(addr, &chaincfg.MainNetParams)
	if err != nil {
		logger.LogDebug(nil, constants.LogDebug,
			"Failed to decode address %s: %v", addr, err)
		return
	}

	categoryMutex.Lock()
	defer categoryMutex.Unlock()

	// logCategorization(addr, decodedAddr)

	switch decodedAddr.(type) {
	case *btcutil.AddressPubKeyHash:
		category.LegacyCount++
	case *btcutil.AddressScriptHash:
		category.SegwitCount++
	case *btcutil.AddressWitnessPubKeyHash:
		category.NativeCount++
	case *btcutil.AddressWitnessScriptHash:
		category.WScriptCount++
	case *btcutil.AddressPubKey:
		category.LegacyCount++
	default:
		// Intentionally left blank for unknown address types
	}
}

func logCategorization(addr string, decodedAddr btcutil.Address) {

	if constants.Logger == nil {
		constants.Logger = log.New(os.Stdout, "", 0)
	}

	if constants.DebugMode {
		switch decodedAddr.(type) {
		case *btcutil.AddressPubKeyHash:
			logger.LogDebug(constants.Logger, constants.LogDebug,
				"Legacy P2PKH:  %s", addr)
		case *btcutil.AddressScriptHash:
			logger.LogDebug(constants.Logger, constants.LogDebug,
				"SegWit P2S):   %s", addr)
		case *btcutil.AddressWitnessPubKeyHash:
			logger.LogDebug(constants.Logger, constants.LogDebug,
				"Native P2WPKH: %s", addr)
		case *btcutil.AddressWitnessScriptHash:
			logger.LogDebug(constants.Logger, constants.LogDebug,
				"Witness P2WSH: %s", addr)
		case *btcutil.AddressPubKey:
			logger.LogDebug(constants.Logger, constants.LogDebug,
				"Public Key:   %s", addr)
		default:
			logger.LogDebug(constants.Logger, constants.LogDebug,
				"Unknown Type: %s", addr)
		}
	}
}
