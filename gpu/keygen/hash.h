#ifndef KEYGEN_HASH_H
#define KEYGEN_HASH_H

#include <cuda_runtime.h>
#include <stdint.h>

// SHA256 functions
__device__ void sha256_init(uint32_t *state);
__device__ void sha256_transform(uint32_t *state, const uint32_t *block);
__device__ void sha256(const unsigned char *input, size_t length,
                       unsigned char *digest);

// RIPEMD160 functions
__device__ void ripemd160_transform(uint32_t *state, const uint32_t *block);
__device__ void ripemd160(const unsigned char *input, size_t length,
                          unsigned char *digest);

// Combined hash function
__device__ void hash160(const unsigned char *input, size_t length,
                        unsigned char *output);

#endif
