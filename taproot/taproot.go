package taproot

import (
	"strings"
)

// bech32 package implementation (simplified)
const charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

func bech32Encode(hrp string, data []byte) (string, error) {
	// Convert data to 5-bit groups
	converted := make([]int, 0, len(data)*8/5+1)
	v := 0
	bits := 0
	for _, value := range data {
		v = (v << 8) | int(value)
		bits += 8
		for bits >= 5 {
			converted = append(converted, (v>>(bits-5))&31)
			bits -= 5
		}
	}
	if bits > 0 {
		converted = append(converted, (v<<(5-bits))&31)
	}

	// Calculate checksum
	chk := bech32Checksum(hrp, converted)
	converted = append(converted, chk...)

	// Encode result
	var ret strings.Builder
	ret.WriteString(hrp)
	ret.WriteByte('1')
	for _, v := range converted {
		ret.WriteByte(charset[v])
	}

	return ret.String(), nil
}

const (
	generator0 uint32 = 0x3b6a57b2
	generator1 uint32 = 0x26508e6d
	generator2 uint32 = 0x1ea119fa
	generator3 uint32 = 0x3d4233dd
	generator4 uint32 = 0x2a1462b3
)

func bech32Checksum(hrp string, data []int) []int {
	chk := 1
	for _, c := range hrp {
		top := chk >> 25
		chk = (chk&0x1ffffff)<<5 ^ int(c)
		for j := 0; j < 5; j++ {
			if (top>>j)&1 == 1 {
				switch j {
				case 0:
					chk ^= int(generator0)
				case 1:
					chk ^= int(generator1)
				case 2:
					chk ^= int(generator2)
				case 3:
					chk ^= int(generator3)
				case 4:
					chk ^= int(generator4)
				}
			}
		}
	}
	chk ^= 1

	for _, v := range data {
		top := chk >> 25
		chk = (chk&0x1ffffff)<<5 ^ v
		for j := 0; j < 5; j++ {
			if (top>>j)&1 == 1 {
				switch j {
				case 0:
					chk ^= int(generator0)
				case 1:
					chk ^= int(generator1)
				case 2:
					chk ^= int(generator2)
				case 3:
					chk ^= int(generator3)
				case 4:
					chk ^= int(generator4)
				}
			}
		}
	}

	ret := make([]int, 6)
	for j := 0; j < 6; j++ {
		ret[j] = (chk >> (5 * (5 - j))) & 31
	}

	return ret
}
