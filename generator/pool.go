package generator

import (
	"crypto/rand"
	"sync"
)

type RNGPool struct {
	pool sync.Pool
	size int
}

func NewRNGPool(size int) *RNGPool {
	return &RNGPool{
		pool: sync.Pool{
			New: func() interface{} {
				return make([]byte, 32) // Restore to original 32 bytes
			},
		},
		size: size,
	}
}

func (p *RNGPool) Get() []byte {
	// Use larger buffer for better entropy
	buf := p.pool.Get().([]byte)
	if _, err := rand.Read(buf); err != nil {
		// If random read fails, return new buffer
		return make([]byte, 32)
	}

	// Copy to prevent reuse
	result := make([]byte, 32)
	copy(result, buf)

	p.pool.Put(buf)
	return result
}
