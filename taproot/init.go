package taproot

import (
	"errors"
	"log"
	"os"

	"github.com/btcsuite/btcd/chaincfg"
	"github.com/btcsuite/btcd/wire"
	"github.com/btcsuite/btcutil/bech32"
)

var logger *log.Logger

func init() {
	logger = log.New(os.Stdout, "", log.LstdFlags)
}

// TaprootAddress
type TaprootAddress struct {
	Addr string
}

func (t *TaprootAddress) String() string {
	return t.Addr
}

func (t *TaprootAddress) EncodeAddress() string {
	return t.Addr
}

func (t *TaprootAddress) ScriptAddress() []byte {
	// Decode the Bech32 address
	hrp, data, err := bech32.Decode(t.Addr)
	if err != nil {
		return nil
	}
	if hrp != "bc" {
		return nil
	}
	// Convert 5-bit groups back to 8-bit bytes
	scriptAddr := make([]byte, 0, len(data)*5/8)
	v := 0
	bits := 0
	for _, value := range data {
		v = (v << 5) | int(value)
		bits += 5
		if bits >= 8 {
			scriptAddr = append(scriptAddr, byte(v>>(bits-8)))
			bits -= 8
		}
	}
	return scriptAddr
}

func (t *TaprootAddress) IsForNet(params *chaincfg.Params) bool {
	return params.Net == wire.MainNet
}

// IsPayToTaproot checks if the given script is a Pay-to-Taproot script.
func IsPayToTaproot(script []byte) bool {
	return len(script) == 34 && script[0] == 0x51 && script[1] == 0x20
}

// ExtractTaprootAddresses extracts Taproot addresses from a script.
func ExtractTaprootAddresses(script []byte) ([]string, error) {
	var addresses []string

	if IsPayToTaproot(script) {
		hash := script[2:]
		address, err := EncodeTaprootAddress(hash)
		if err == nil {
			addresses = append(addresses, address)
		}
	}

	return addresses, nil
}

// EncodeTaprootAddress encodes the given 32-byte hash into a Taproot address.
func EncodeTaprootAddress(hash []byte) (string, error) {
	if len(hash) != 32 {
		return "", errors.New("invalid hash length for Taproot address")
	}

	// Encode the hash with Bech32 encoding
	hrp := "bc"
	data := append([]byte{0}, hash...)
	encoded, err := bech32.Encode(hrp, data)
	if err != nil {
		return "", err
	}

	return encoded, nil
}

// Helper function for Bech32 decoding
func bech32Decode(bech string) ([]byte, error) {
	hrp, data, err := bech32.Decode(bech)
	if err != nil {
		return nil, err
	}
	if hrp != "bc" {
		return nil, errors.New("invalid HRP")
	}
	// Convert 5-bit groups back to 8-bit bytes
	ret := make([]byte, 0, len(data)*5/8)
	v := 0
	bits := 0
	for _, value := range data {
		v = (v << 5) | int(value)
		bits += 5
		if bits >= 8 {
			ret = append(ret, byte(v>>(bits-8)))
			bits -= 8
		}
	}
	return ret, nil
}
