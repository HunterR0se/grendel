
#include "brain_secp.h"
#include "constants.h"
#include "group.h"
// #include <stdio.h>

__device__ void point_add_mixed(ProjectivePoint *R, const uint32_t *X2,
                                const uint32_t *Y2) {
  // Mixed addition formula (adding a point in projective coordinates to one in
  // affine) This is faster than full projective addition because Z2 = 1

  if (IsZero(R->Z)) {
    memcpy(R->X, X2, 32);
    memcpy(R->Y, Y2, 32);
    R->Z[0] = 1;
    return;
  }

  uint32_t Z1Z1[8]; // Z1^2
  uint32_t U2[8];   // X2*Z1Z1
  uint32_t S2[8];   // Y2*Z1*Z1Z1
  uint32_t H[8];    // U2-X1
  uint32_t HH[8];   // H^2
  uint32_t I[8];    // 4*HH
  uint32_t J[8];    // H*I
  uint32_t r[8];    // 2*(S2-Y1)
  uint32_t V[8];    // X1*I

  // Z1Z1 = Z1^2
  MulModP(Z1Z1, R->Z, R->Z);

  // U2 = X2*Z1Z1
  MulModP(U2, X2, Z1Z1);

  // S2 = Y2*Z1*Z1Z1
  MulModP(S2, Z1Z1, R->Z);
  MulModP(S2, S2, Y2);

  // H = U2-X1
  Sub(H, U2, R->X);
  ModP(H);

  // HH = H^2
  MulModP(HH, H, H);

  // I = 4*HH
  Add(I, HH, HH);
  Add(I, I, I);
  ModP(I);

  // J = H*I
  MulModP(J, H, I);

  // r = 2*(S2-Y1)
  Sub(r, S2, R->Y);
  Add(r, r, r);
  ModP(r);

  // V = X1*I
  MulModP(V, R->X, I);

  // X3 = r^2 - J - 2*V
  MulModP(R->X, r, r);
  Sub(R->X, R->X, J);
  Sub(R->X, R->X, V);
  Sub(R->X, R->X, V);
  ModP(R->X);

  // Y3 = r*(V-X3) - 2*Y1*J
  Sub(V, V, R->X);
  MulModP(R->Y, r, V);
  MulModP(V, R->Y, J);
  Sub(R->Y, R->Y, V);
  Sub(R->Y, R->Y, V);
  ModP(R->Y);

  // Z3 = 2*Z1*H
  MulModP(R->Z, R->Z, H);
  Add(R->Z, R->Z, R->Z);
  ModP(R->Z);
}

__device__ void point_double(ProjectivePoint *P, int tid) {
  // If point is infinity, nothing to do
  if (IsZero(P->Z))
    return;

  // Use registers for faster access
  uint32_t *X = P->X;
  uint32_t *Y = P->Y;
  uint32_t *Z = P->Z;

  // Minimize stack usage with register variables
  uint32_t XX[8], YY[8], ZZ[8];
  uint32_t M[8], S[8];

  // XX = X² (in Montgomery form)
  MulModP(XX, X, X);

  // YY = Y²
  MulModP(YY, Y, Y);

  // ZZ = Z²
  MulModP(ZZ, Z, Z);

  // M = 3*XX - a*ZZ² = 3*XX (since a = 0)
  uint32_t t1[8];
  Add(M, XX, XX); // 2*XX
  Add(M, M, XX);  // 3*XX
  ModP(M);

  // S = 4*X*YY
  MulModP(S, X, YY);
  Add(S, S, S);
  ModP(S);
  Add(S, S, S);
  ModP(S);

  // X₃ = M² - 2*S
  MulModP(t1, M, M);
  Sub(t1, t1, S);
  Sub(X, t1, S);
  ModP(X);

  // Y₃ = M*(S - X₃) - 8*YY²
  MulModP(YY, YY, YY); // YY = YY²
  Sub(t1, S, X);
  MulModP(Y, M, t1);
  Add(t1, YY, YY); // 2*YY²
  Add(t1, t1, t1); // 4*YY²
  Add(t1, t1, t1); // 8*YY²
  Sub(Y, Y, t1);
  ModP(Y);

  // Z₃ = 2*Y*Z
  MulModP(Z, Y, Z);
  Add(Z, Z, Z);
  ModP(Z);
}

__device__ void point_to_affine(ProjectivePoint *P, uint32_t *x, uint32_t *y) {
  if (IsZero(P->Z)) {
    memset(x, 0, 32);
    memset(y, 0, 32);
    return;
  }

  // Calculate Z inverse using Fermat's little theorem: Z^(-1) = Z^(p-2) mod p
  uint32_t z_inv[8];
  uint32_t z_inv_squared[8];

  // Use binary exponentiation for z_inv calculation
  uint32_t base[8];
  memcpy(base, P->Z, 32);
  uint32_t result[8] = {1, 0, 0, 0, 0, 0, 0, 0};

// p-2 has a special form for our prime, use it
#pragma unroll
  for (int i = 255; i >= 0; i--) {
    MulModP(result, result, result);
    if (i == 255 || i == 254 || i == 253 || i == 252 || // top bits
        i == 251 || i == 250 || i == 249 || i == 248 || // are all 1
        i == 247 || i == 246 || i == 245 || i == 244) {
      MulModP(result, result, base);
    }
  }

  memcpy(z_inv, result, 32);

  // Calculate x = X * Z^(-1)
  MulModP(x, P->X, z_inv);

  // Calculate y = Y * Z^(-1)
  MulModP(y, P->Y, z_inv);
}

__device__ void secp256k1_scalar_multiply_base(uint32_t *result_x,
                                               uint32_t *result_y,
                                               const uint32_t *scalar) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Use registers for accumulator
  ProjectivePoint R;
  memset(&R, 0, sizeof(ProjectivePoint));

  // Pre-compute first window lookup to avoid branching in loop
  uint32_t window = (scalar[7] >> 28) & 0xF;
  if (window) {
    memcpy(&R, &PRECOMP_G[window], sizeof(ProjectivePoint));
  }

// Process 4 bits at a time from MSB to LSB
#pragma unroll 1
  for (int word = 7; word >= 0; word--) {
    uint32_t bits = scalar[word];
#pragma unroll 8
    for (int shift = 28; shift >= 0; shift -= 4) {
      if (word != 7 ||
          shift != 28) { // Skip first window which we already processed
// Always double 4 times
#pragma unroll 4
        for (int j = 0; j < 4; j++) {
          point_double(&R, tid);
        }

        window = (bits >> shift) & 0xF;
        if (window) {
          point_add_mixed(&R, PRECOMP_G[window].X, PRECOMP_G[window].Y);
        }
      }
    }
  }

  // Convert result to affine coordinates
  point_to_affine(&R, result_x, result_y);
}

__device__ void _secp256k1_scalar_multiply_base(uint32_t *result_x,
                                                uint32_t *result_y,
                                                const uint32_t *scalar) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Convert scalar from big-endian to little-endian for processing
  uint32_t scalar_le[8];
  for (int i = 0; i < 8; i++) {
    scalar_le[i] = scalar[7 - i];
  }

  uint32_t rx[8] = {0}, ry[8] = {0};
  bool found_one = false;

  // Process bits from MSB to LSB
  for (int i = 255; i >= 0; i--) {
    int word = i >> 5; // Word index in little-endian order
    int bit = i & 31;  // Bit position in word

    if (found_one) {
      SecpDblStep(rx, ry, tid);
    }

    if ((scalar_le[word] >> bit) & 1) {
      if (!found_one) {
        memcpy(rx, G_x, sizeof(uint32_t) * 8);
        memcpy(ry, G_y, sizeof(uint32_t) * 8);
        found_one = true;
      } else {
        SecpAddStep(rx, ry, G_x, G_y, tid);
      }
    }
  }

  // Convert result from little-endian to big-endian
  for (int i = 0; i < 8; i++) {
    result_x[i] = rx[7 - i];
    result_y[i] = ry[7 - i];
  }
}
