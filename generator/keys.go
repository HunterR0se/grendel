package generator

import (
	"Grendel/constants"
	"math/rand"

	"github.com/btcsuite/btcd/btcec/v2"
	"github.com/btcsuite/btcd/chaincfg"
	"github.com/btcsuite/btcutil"
)

// GlobalAddressTypeStats holds the statistics for different types of Bitcoin addresses.
type GlobalAddressTypeStats struct {
	Legacy int
	Segwit int
	Native int
}

var globalStats = &Stats{} // Initialize with zero values

func DecrementAddressType(addrType AddressType) {
	constants.GlobalStats.Lock()
	defer constants.GlobalStats.Unlock()

	switch addrType {
	case Legacy:
		constants.GlobalStats.LegacyCount--
	case Segwit:
		constants.GlobalStats.SegwitCount--
	case Native:
		constants.GlobalStats.NativeCount--
	}
}

// getGlobalAddressTypeStats retrieves the global statistics for Bitcoin address types.
//
// It uses the globalStats directly, assuming they are updated in the application
func getGlobalAddressTypeStats() GlobalAddressTypeStats {
	if globalStats == nil {
		// Handle the case where globalStats is not initialized
		return GlobalAddressTypeStats{}
	}

	return GlobalAddressTypeStats{
		Legacy: int(globalStats.LegacyCount),
		Segwit: int(globalStats.SegwitCount),
		Native: int(globalStats.NativeCount),
	}
}

func GetStats() Stats {
	return *globalStats
}

func GenerateRawPrivateKey(seed []byte, stats *Stats) (*btcec.PrivateKey, string, AddressType, error) {
	privKey, _ := btcec.PrivKeyFromBytes(seed)
	pubKey := privKey.PubKey()
	pubKeyBytes := pubKey.SerializeCompressed()

	r := rand.Float64()
	var addr btcutil.Address
	var err error
	var addrType AddressType

	if r < 0.33 {
		addr, err = btcutil.NewAddressPubKeyHash(btcutil.Hash160(pubKeyBytes), &chaincfg.MainNetParams)
		addrType = Legacy
		constants.GlobalStats.Lock()
		constants.GlobalStats.LegacyCount++
		constants.GlobalStats.Unlock()
	} else if r < 0.66 {
		addr, err = btcutil.NewAddressScriptHashFromHash(btcutil.Hash160(pubKeyBytes), &chaincfg.MainNetParams)
		addrType = Segwit
		constants.GlobalStats.Lock()
		constants.GlobalStats.SegwitCount++
		constants.GlobalStats.Unlock()
	} else {
		addr, err = btcutil.NewAddressWitnessPubKeyHash(btcutil.Hash160(pubKeyBytes), &chaincfg.MainNetParams)
		addrType = Native
		constants.GlobalStats.Lock()
		constants.GlobalStats.NativeCount++
		constants.GlobalStats.Unlock()
	}

	if err != nil {
		return nil, "", addrType, err
	}

	constants.IncrementGenerated()
	return privKey, addr.EncodeAddress(), addrType, nil
}
