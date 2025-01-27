#ifndef SECP256K1_CONSTANTS_H
#define SECP256K1_CONSTANTS_H

#include "group.h" // Add this to get ProjectivePoint definition
#include <cuda_runtime.h>
#include <stdint.h>

// Declare constants as extern
extern __device__ __constant__ uint32_t G_x[8];
extern __device__ __constant__ uint32_t G_y[8];
extern __device__ __constant__ uint32_t P[8];
extern __device__ __constant__ uint32_t N[8];
extern __device__ __constant__ struct ProjectivePoint PRECOMP_G[16];

#endif
