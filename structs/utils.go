package structs

import "math"

// encodeHeapItem combines a distance (float32) and an ID (int) into a single uint64.
// The encoding preserves the ordering of distances by manipulating the float32 bits
// in a way that maintains correct ordering for both positive and negative numbers.
// For negative numbers, the bits are inverted to ensure they sort correctly when
// compared as uint64.
func EncodeHeapItem(dist float32, id int) uint64 {
	if id < 0 {
		panic("ID must be non-negative")
	}
	if id > math.MaxInt32 {
		panic("ID must not exceed MaxInt32")
	}

	bits := math.Float32bits(dist)
	// Invert the bits if it's a negative number to maintain correct ordering
	if (bits & 0x80000000) != 0 {
		bits = ^bits
	} else {
		bits = bits ^ 0x80000000
	}
	return (uint64(bits) << 32) | uint64(id)
}

// decodeHeapItem extracts the original distance and ID from an encoded uint64 value.
func DecodeHeapItem(item uint64) (float32, int) {
	bits := uint32(item >> 32)
	// Reverse the bit manipulation from encoding
	if (bits & 0x80000000) == 0 {
		bits = ^bits
	} else {
		bits = bits ^ 0x80000000
	}
	return math.Float32frombits(bits), int(item & 0xffffffff)
}
