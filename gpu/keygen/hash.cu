#include "hash.h"
#include <stdio.h>

// SHA256 Constants
__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

__device__ uint32_t rotr(uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}

__device__ void sha256_transform(uint32_t *state, const uint32_t *block) {
  uint32_t w[64];
  uint32_t a, b, c, d, e, f, g, h;
  uint32_t t1, t2;

// Copy block into w
#pragma unroll 16
  for (int i = 0; i < 16; i++) {
    w[i] = block[i];
  }

// Extend the sixteen 32-bit words into sixty-four 32-bit words
#pragma unroll 48
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }

  // Initialize hash value for this chunk
  a = state[0];
  b = state[1];
  c = state[2];
  d = state[3];
  e = state[4];
  f = state[5];
  g = state[6];
  h = state[7];

// Main loop
#pragma unroll 64
  for (int i = 0; i < 64; i++) {
    t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) +
         K[i] + w[i];
    t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) +
         ((a & b) ^ (a & c) ^ (b & c));
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  // Add chunk's hash to result
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

__device__ void sha256_init(uint32_t *state) {
  state[0] = 0x6a09e667;
  state[1] = 0xbb67ae85;
  state[2] = 0x3c6ef372;
  state[3] = 0xa54ff53a;
  state[4] = 0x510e527f;
  state[5] = 0x9b05688c;
  state[6] = 0x1f83d9ab;
  state[7] = 0x5be0cd19;
}

__device__ void sha256(const unsigned char *input, size_t length,
                       unsigned char *digest) {

  if (length > 1024) { // Add reasonable maximum
    printf("Input length too large: %lu\n", (unsigned long)length);
    return;
  }

  uint32_t state[8];
  uint32_t block[16];
  unsigned int bytesProcessed = 0;
  uint64_t totalBits = length * 8;

  sha256_init(state);

  // Process all complete blocks
  while (bytesProcessed + 64 <= length) {

// Copy and convert current block
#pragma unroll 16
    for (int i = 0; i < 16; i++) {
      block[i] = ((uint32_t)input[bytesProcessed + i * 4] << 24) |
                 ((uint32_t)input[bytesProcessed + i * 4 + 1] << 16) |
                 ((uint32_t)input[bytesProcessed + i * 4 + 2] << 8) |
                 ((uint32_t)input[bytesProcessed + i * 4 + 3]);
    }
    sha256_transform(state, block);
    bytesProcessed += 64;
  }

  // Final block with padding
  memset(block, 0, sizeof(block));
  int remainingBytes = length - bytesProcessed;

  // Copy remaining bytes
  for (int i = 0; i < remainingBytes; i++) {
    int idx = i >> 2;
    int shift = (3 - (i & 3)) << 3;
    block[idx] |= (uint32_t)input[bytesProcessed + i] << shift;
  }

  // Add padding bit
  int idx = remainingBytes >> 2;
  int shift = (3 - (remainingBytes & 3)) << 3;
  block[idx] |= (uint32_t)0x80 << shift;

  // Add length if it fits, otherwise process this block and create a new one
  if (remainingBytes >= 56) {
    sha256_transform(state, block);
    memset(block, 0, sizeof(block));
  }

  // Add total length
  block[14] = (uint32_t)(totalBits >> 32);
  block[15] = (uint32_t)totalBits;

  sha256_transform(state, block);

// Copy result to digest
#pragma unroll 8
  for (int i = 0; i < 8; i++) {
    digest[i * 4] = (state[i] >> 24) & 0xFF;
    digest[i * 4 + 1] = (state[i] >> 16) & 0xFF;
    digest[i * 4 + 2] = (state[i] >> 8) & 0xFF;
    digest[i * 4 + 3] = state[i] & 0xFF;
  }
}

// RIPEMD160 Constants
__device__ __constant__ uint32_t RK[5][10] = {
    {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E, 0x50A28BE6,
     0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000},
    {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000, 0x00000000,
     0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E}};

__device__ __constant__ uint32_t RR[5][10] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {7, 4, 13, 1, 10, 6, 15, 3, 12, 0}};

__device__ __constant__ uint32_t RS[5][10] = {
    {11, 14, 15, 12, 5, 8, 7, 9, 11, 13}, {7, 6, 8, 13, 11, 9, 7, 15, 7, 12}};

__device__ uint32_t ripemd_f(int j, uint32_t x, uint32_t y, uint32_t z) {
  if (j < 16)
    return x ^ y ^ z;
  if (j < 32)
    return (x & y) | (~x & z);
  if (j < 48)
    return (x | ~y) ^ z;
  if (j < 64)
    return (x & z) | (y & ~z);
  return x ^ (y | ~z);
}

__device__ uint32_t ripemd_rol(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

__device__ void ripemd160_transform(uint32_t *state, const uint32_t *block) {
  uint32_t a1 = state[0], b1 = state[1], c1 = state[2], d1 = state[3],
           e1 = state[4];
  uint32_t a2 = state[0], b2 = state[1], c2 = state[2], d2 = state[3],
           e2 = state[4];
  uint32_t t;

#pragma unroll 80
  for (int j = 0; j < 80; j++) {
    int round = j >> 4;
    t = ripemd_rol(a1 + ripemd_f(j, b1, c1, d1) + block[RR[0][j % 10]] +
                       RK[0][round],
                   RS[0][j % 10]) +
        e1;
    a1 = e1;
    e1 = d1;
    d1 = ripemd_rol(c1, 10);
    c1 = b1;
    b1 = t;

    t = ripemd_rol(a2 + ripemd_f(79 - j, b2, c2, d2) + block[RR[1][j % 10]] +
                       RK[1][round],
                   RS[1][j % 10]) +
        e2;
    a2 = e2;
    e2 = d2;
    d2 = ripemd_rol(c2, 10);
    c2 = b2;
    b2 = t;
  }

  t = state[1] + c1 + d2;
  state[1] = state[2] + d1 + e2;
  state[2] = state[3] + e1 + a2;
  state[3] = state[4] + a1 + b2;
  state[4] = state[0] + b1 + c2;
  state[0] = t;
}

__device__ void ripemd160(const unsigned char *input, size_t length,
                          unsigned char *digest) {
  uint32_t state[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476,
                       0xC3D2E1F0};
  uint32_t block[16];
  unsigned int bytesProcessed = 0;
  uint64_t totalBits = length * 8;

  // Process all complete blocks
  while (bytesProcessed + 64 <= length) {
#pragma unroll 16
    for (int i = 0; i < 16; i++) {
      block[i] = ((uint32_t)input[bytesProcessed + i * 4] << 24) |
                 ((uint32_t)input[bytesProcessed + i * 4 + 1] << 16) |
                 ((uint32_t)input[bytesProcessed + i * 4 + 2] << 8) |
                 ((uint32_t)input[bytesProcessed + i * 4 + 3]);
    }
    ripemd160_transform(state, block);
    bytesProcessed += 64;
  }

  // Final block with padding
  memset(block, 0, sizeof(block));
  int remainingBytes = length - bytesProcessed;

  // Copy remaining bytes
  for (int i = 0; i < remainingBytes; i++) {
    int idx = i >> 2;
    int shift = (3 - (i & 3)) << 3;
    block[idx] |= (uint32_t)input[bytesProcessed + i] << shift;
  }

  // Add padding bit
  int idx = remainingBytes >> 2;
  int shift = (3 - (remainingBytes & 3)) << 3;
  block[idx] |= (uint32_t)0x80 << shift;

  // Add length if it fits, otherwise process this block and create a new one
  if (remainingBytes >= 56) {
    ripemd160_transform(state, block);
    memset(block, 0, sizeof(block));
  }

  // Add total length
  block[14] = (uint32_t)(totalBits);
  block[15] = (uint32_t)(totalBits >> 32);

  ripemd160_transform(state, block);

  // Copy result to digest
  for (int i = 0; i < 5; i++) {
    digest[i * 4] = state[i] & 0xFF;
    digest[i * 4 + 1] = (state[i] >> 8) & 0xFF;
    digest[i * 4 + 2] = (state[i] >> 16) & 0xFF;
    digest[i * 4 + 3] = (state[i] >> 24) & 0xFF;
  }
}

__device__ void hash160(const unsigned char *input, size_t length,
                        unsigned char *output) {
  unsigned char sha256_output[32];

  // First round: SHA256
  sha256(input, length, sha256_output);

  // Second round: RIPEMD160
  ripemd160(sha256_output, 32, output);
}
