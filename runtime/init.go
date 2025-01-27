package runtime

import (
	"Grendel/generator"
	"Grendel/gpu"
	"Grendel/loader"
	"Grendel/parser"
	"Grendel/utils"
	"log"
	"sync"

	"github.com/btcsuite/btcd/btcec/v2"
)

var CloseOnce sync.Once

type WalletInfo struct {
	Address    string
	PrivateKey *btcec.PrivateKey
	AddrType   generator.AddressType
	// Seed       string // Uncomment if/when we add seed phrase support
}

type AppContext struct {
	LocalLog     *log.Logger
	GenMode      bool
	DebugMode    bool
	ForceReparse bool
	CpuMode      bool
	GpuMode      bool
	BaseDir      string
	ParserPath   string
	DbPath       string
	AddressPath  string
	Parser       *parser.Parser
	ShutdownChan chan struct{}

	AddressChan     chan *WalletInfo
	DoneChan        chan struct{}
	MemLimits       *utils.MemoryLimits
	BlockLoader     *loader.BlockLoader
	Wg              sync.WaitGroup
	CloseOnce       sync.Once
	BlockLoadDone   chan struct{}
	AddressesLoaded bool
	// GPU!
	GpuInfo   gpu.GPUInfo
	Generator gpu.Generator
}
