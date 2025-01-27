#ifndef BRAIN_SECP_H
#define BRAIN_SECP_H

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {
// Core functions we're extracting
__device__ void ScalarMulP(uint32_t *result_x, uint32_t *result_y,
                           const uint32_t *scalar);
__device__ void InitializeSecp(void);

// Fixed function declarations - changed Add to void
__device__ void SetBigInt(uint32_t *r, const uint32_t *a);
__device__ int IsEqual(const uint32_t *a, const uint32_t *b);
__device__ int IsZero(const uint32_t *a);
__device__ uint64_t Add(uint32_t *r, const uint32_t *a,
                        const uint32_t *b); // Return carry
__device__ void Sub(uint32_t *r, const uint32_t *a, const uint32_t *b);
__device__ void Mult(uint32_t *r, const uint32_t *a, const uint32_t *b);
__device__ void ModP(uint32_t *r);
__device__ void MulModP(uint32_t *r, const uint32_t *a, const uint32_t *b);
__device__ void ModInv(uint32_t *r, const uint32_t *x);

__device__ void SecpDblStep(uint32_t *rx, uint32_t *ry, int tid);
__device__ void SecpAddStep(uint32_t *rx, uint32_t *ry, const uint32_t *px,
                            const uint32_t *py, int tid);
}

// Constants needed for their implementation
__device__ __constant__ extern uint32_t _INC[8];
__device__ __constant__ extern uint32_t _EC[8];

#endif
