package generator

import "github.com/btcsuite/btcd/btcec/v2"

type AddressType int

const (
	Legacy AddressType = iota
	Segwit
	Native
)

type Stats struct {
	LegacyCount uint64
	SegwitCount uint64
	NativeCount uint64
	Generated   uint64
	Found       uint64
}

type Generator interface {
	Generate(count int) ([]*btcec.PrivateKey, []string, []AddressType, error)
	GetStats() *Stats // Make sure this is implemented
	Close() error
}
