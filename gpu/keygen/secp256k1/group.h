#ifndef SECP256K1_GROUP_H
#define SECP256K1_GROUP_H

#include <cuda_runtime.h>
#include <stdint.h>

struct ProjectivePoint {
  uint32_t X[8];
  uint32_t Y[8];
  uint32_t Z[8];
};

// expose the scalar multiply function
__device__ void secp256k1_scalar_multiply_base(uint32_t *result_x,
                                               uint32_t *result_y,
                                               const uint32_t *scalar);

#endif
