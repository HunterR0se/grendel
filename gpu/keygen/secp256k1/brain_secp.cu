#include "brain_secp.h"
#include "constants.h" // Add this include
// #include "stdio.h"

#ifdef DEBUG_MODE
#define SECP_LOG(tid, fmt, ...)                                                \
  printf("[SECP][Thread %d] " fmt "\n", tid, ##__VA_ARGS__)
#else
#define SECP_LOG(tid, fmt, ...)
#endif

#define SECP_ERROR(tid, fmt, ...)                                              \
  printf("[SECP ERROR][Thread %d] " fmt "\n", tid, ##__VA_ARGS__)

__device__ int IsGreaterOrEqual(const uint32_t *a, const uint32_t *b) {
  for (int i = 7; i >= 0; i--) {
    if (a[i] > b[i])
      return 1;
    if (a[i] < b[i])
      return 0;
  }
  return 1;
}

__device__ void ModP(uint32_t *r) {
  // First try fast reduction
  uint32_t carry = 0;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint64_t t = (uint64_t)r[i] + carry;
    r[i] = (uint32_t)t;
    carry = t >> 32;
  }

  if (carry || IsGreaterOrEqual(r, _EC)) {
    carry = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      uint64_t t = (uint64_t)r[i] - _EC[i] - carry;
      r[i] = (uint32_t)t;
      carry = -(t >> 32);
    }
  }
}

__device__ uint64_t Add(uint32_t *r, const uint32_t *a, const uint32_t *b) {
  uint64_t carry = 0;

  for (int i = 0; i < 8; i++) {
    carry += (uint64_t)a[i] + b[i];
    r[i] = (uint32_t)carry;
    carry >>= 32;
  }
  return carry; // Return the final carry bit
}

__device__ void Sub(uint32_t *r, const uint32_t *a, const uint32_t *b) {
  int64_t borrow = 0;

  for (int i = 0; i < 8; i++) {
    int64_t diff = (int64_t)a[i] - b[i] - (borrow & 1);
    r[i] = (uint32_t)diff;
    borrow = diff >> 63; // Arithmetic shift preserves sign
  }
}

__device__ void SetBigInt(uint32_t *r, const uint32_t *a) {
  for (int i = 0; i < 8; i++) {
    r[i] = a[i];
  }
}

__device__ int IsEqual(const uint32_t *a, const uint32_t *b) {
  for (int i = 0; i < 8; i++) {
    if (a[i] != b[i]) {
      return 0;
    }
  }
  return 1;
}

__device__ int IsZero(const uint32_t *a) {
  for (int i = 0; i < 8; i++) {
    if (a[i] != 0) {
      return 0;
    }
  }
  return 1;
}

__device__ void Mult(uint32_t *r, const uint32_t *a, const uint32_t *b) {
  uint32_t t[16] = {0};
  for (int i = 0; i < 8; i++) {
    uint64_t carry = 0;
    for (int j = 0; j < 8; j++) {
      carry += (uint64_t)t[i + j] + ((uint64_t)a[i] * b[j]);
      t[i + j] = (uint32_t)carry;
      carry >>= 32;
    }
    t[i + 8] = (uint32_t)carry;
  }

  // Reduction mod P
  uint64_t carry = 0;
  for (int i = 0; i < 8; i++) {
    carry += t[i];
    r[i] = (uint32_t)carry;
    carry >>= 32;
  }
}

__device__ void _SecpAddStep(uint32_t *rx, uint32_t *ry, const uint32_t *px,
                             const uint32_t *py, int tid) {
  uint64_t start = clock64();

  // Handle identity points
  if (IsZero(rx)) {
    SetBigInt(rx, px);
    SetBigInt(ry, py);
    return;
  }
  if (IsZero(px)) {
    return;
  }
  uint64_t ident = clock64();

  uint32_t s[8];   // Slope
  uint32_t tmp[8]; // Temporary for calculations
  uint32_t xr[8];  // New x coordinate
  uint32_t yr[8];  // New y coordinate
  uint32_t dx[8];  // x2 - x1

  // Check if points are the same
  if (IsEqual(rx, px)) {
    if (!IsEqual(ry, py)) {
      // Set result to identity point
      for (int i = 0; i < 8; i++) {
        rx[i] = 0;
        ry[i] = 0;
      }
      return;
    }
    SecpDblStep(rx, ry, tid);
    return;
  }
  uint64_t equal = clock64();

  // Calculate slope = (y2 - y1)/(x2 - x1)
  Sub(s, py, ry);  // y2 - y1
  Sub(dx, px, rx); // x2 - x1
  uint64_t sub = clock64();

  ModInv(dx, dx); // 1/(x2 - x1)
  uint64_t inv = clock64();

  MulModP(s, s, dx); // (y2 - y1)/(x2 - x1)
  uint64_t slope = clock64();

  // Calculate xr = s² - x1 - x2
  MulModP(xr, s, s); // s²
  Sub(xr, xr, rx);   // s² - x1
  Sub(xr, xr, px);   // s² - x1 - x2
  ModP(xr);
  uint64_t xcoord = clock64();

  // Calculate yr = s(x1 - xr) - y1
  Sub(tmp, rx, xr);     // x1 - xr
  MulModP(tmp, s, tmp); // s(x1 - xr)
  Sub(yr, tmp, ry);     // s(x1 - xr) - y1
  ModP(yr);
  uint64_t ycoord = clock64();

  // Copy results back
  SetBigInt(rx, xr);
  SetBigInt(ry, yr);
  uint64_t end = clock64();

  /*
  if (tid == 0) {
    printf("AddStep: Ident=%lu Equal=%lu Sub=%lu Inv=%lu Slope=%lu Xcoord=%lu "
           "Ycoord=%lu Copy=%lu\n",
           ident - start, equal - ident, sub - equal, inv - sub, slope - inv,
           xcoord - slope, ycoord - xcoord, end - ycoord);
  }
  */
}

__device__ void SecpAddStep(uint32_t *rx, uint32_t *ry, const uint32_t *px,
                            const uint32_t *py, int tid) {
  // Handle special cases with direct returns
  if (IsZero(rx)) {
    SetBigInt(rx, px);
    SetBigInt(ry, py);
    return;
  }
  if (IsZero(px))
    return;

  // Check if points are the same
  if (IsEqual(rx, px)) {
    if (!IsEqual(ry, py)) {
      memset(rx, 0, 8 * sizeof(uint32_t));
      memset(ry, 0, 8 * sizeof(uint32_t));
      return;
    }
    SecpDblStep(rx, ry, tid);
    return;
  }

  // Calculate slope using registers
  uint32_t s[8], dx[8], tmp[8];

#pragma unroll
  for (int i = 0; i < 8; i++) {
    dx[i] = px[i];
    s[i] = py[i];
  }

  // Compute x2-x1 and y2-y1 in parallel
  uint64_t borrow1 = 0, borrow2 = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint64_t diff1 = (uint64_t)dx[i] - rx[i] - borrow1;
    uint64_t diff2 = (uint64_t)s[i] - ry[i] - borrow2;
    dx[i] = (uint32_t)diff1;
    s[i] = (uint32_t)diff2;
    borrow1 = -(diff1 >> 32);
    borrow2 = -(diff2 >> 32);
  }

  // Compute inverse and multiplication in one step
  ModInv(dx, dx);
  MulModP(s, s, dx);

  // Calculate new x coordinate
  MulModP(tmp, s, s);
  Sub(tmp, tmp, rx);
  Sub(tmp, tmp, px);
  ModP(tmp);

  // Calculate new y coordinate
  Sub(dx, rx, tmp);
  MulModP(dx, s, dx);
  Sub(ry, dx, ry);
  ModP(ry);

  // Set new x coordinate
  SetBigInt(rx, tmp);
}

// --------------------------------------

__device__ void SecpDblStep(uint32_t *rx, uint32_t *ry, int tid) {
  uint64_t start = clock64();

  // Check for identity point
  if (IsZero(rx)) {
    return;
  }

  uint32_t s[8] = {0};   // Slope
  uint32_t tmp[8] = {0}; // Temporary storage
  uint32_t xr[8] = {0};  // New x coordinate
  uint32_t yr[8] = {0};  // New y coordinate

  // Calculate x²
  MulModP(tmp, rx, rx);
  uint64_t mul = clock64();

  // Multiply by 3
  uint32_t three[8] = {3, 0, 0, 0, 0, 0, 0, 0};
  MulModP(s, tmp, three);
  uint64_t mul3 = clock64();

  // Calculate 2y
  uint32_t two[8] = {2, 0, 0, 0, 0, 0, 0, 0};
  MulModP(tmp, ry, two);
  uint64_t mul2 = clock64();

  // Compute inverse of 2y
  ModInv(tmp, tmp);
  uint64_t inv = clock64();

  // Rest of function...
  SetBigInt(rx, xr);
  SetBigInt(ry, yr);

  /*
  if (tid == 0) {
    printf("DblStep: Mul=%lu Mul3=%lu Mul2=%lu Inv=%lu\n", mul - start,
           mul3 - mul, mul2 - mul3, inv - mul2);
  }
  */
}

// ---------------------------------------

__device__ void MulModP(uint32_t *r, const uint32_t *a, const uint32_t *b) {
  // Use registers for better performance
  uint32_t t[8] = {0};
  uint64_t c = 0;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint64_t carry = 0;
    for (int j = 0; j <= i; j++) {
      carry += (uint64_t)a[j] * b[i - j] + t[i];
    }
    t[i] = (uint32_t)carry;
    c += carry >> 32;
  }

#pragma unroll
  for (int i = 8; i < 16; i++) {
    uint64_t carry = c;
    c = 0;
    int j = i - 7;
#pragma unroll
    for (; j < 8; j++) {
      carry += (uint64_t)a[j] * b[i - j];
    }
    t[i - 8] = (uint32_t)carry;
    c = carry >> 32;
  }

  // Fast modular reduction
  uint64_t carry = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    carry += t[i];
    r[i] = (uint32_t)carry;
    carry >>= 32;
  }

  if (carry || IsGreaterOrEqual(r, _EC)) {
    carry = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      uint64_t tmp = (uint64_t)r[i] - _EC[i] - carry;
      r[i] = (uint32_t)tmp;
      carry = -(tmp >> 63);
    }
  }
}

__device__ void _MulModP(uint32_t *r, const uint32_t *a, const uint32_t *b) {
  uint32_t t[16] = {0};

  // Schoolbook multiplication
  for (int i = 0; i < 8; i++) {
    uint64_t carry = 0;
    for (int j = 0; j < 8; j++) {
      carry += (uint64_t)t[i + j] + ((uint64_t)a[i] * b[j]);
      t[i + j] = (uint32_t)carry;
      carry >>= 32;
    }
    t[i + 8] = (uint32_t)carry;
  }

  // Reduction modulo P
  uint64_t carry = 0;
  for (int i = 0; i < 8; i++) {
    carry += t[i];
    r[i] = (uint32_t)carry;
    carry >>= 32;
  }

  ModP(r);
}

__device__ void ModInv(uint32_t *r, const uint32_t *x) {
  uint32_t u[8], v[8], x1[8], x2[8];

  // Initialize
  SetBigInt(u, x);
  SetBigInt(v, _EC);
  memset(x1, 0, sizeof(x1));
  x1[0] = 1;
  memset(x2, 0, sizeof(x2));

  while (!IsZero(u)) {
    while (!(u[0] & 1)) { // u is even
      for (int i = 0; i < 7; i++) {
        u[i] = (u[i] >> 1) | (u[i + 1] << 31);
      }
      u[7] >>= 1;

      if (x1[0] & 1) { // x1 is odd
        uint64_t carry = Add(x1, x1, _EC);
      }
      for (int i = 0; i < 7; i++) {
        x1[i] = (x1[i] >> 1) | (x1[i + 1] << 31);
      }
      x1[7] >>= 1;
    }

    while (!(v[0] & 1)) { // v is even
      for (int i = 0; i < 7; i++) {
        v[i] = (v[i] >> 1) | (v[i + 1] << 31);
      }
      v[7] >>= 1;

      if (x2[0] & 1) { // x2 is odd
        uint64_t carry = Add(x2, x2, _EC);
      }
      for (int i = 0; i < 7; i++) {
        x2[i] = (x2[i] >> 1) | (x2[i + 1] << 31);
      }
      x2[7] >>= 1;
    }

    if (IsGreaterOrEqual(u, v)) {
      Sub(u, u, v);
      Sub(x1, x1, x2);
      if (IsGreaterOrEqual(x1, _EC)) {
        Add(x1, x1, _EC);
      }
    } else {
      Sub(v, v, u);
      Sub(x2, x2, x1);
      if (IsGreaterOrEqual(x2, _EC)) {
        Add(x2, x2, _EC);
      }
    }
  }

  SetBigInt(r, x2);
}

__device__ void _ModInv(uint32_t *r, const uint32_t *x) {
  uint32_t exp[8];
  SetBigInt(exp, _EC);
  Sub(exp, exp, _INC);
  Sub(exp, exp, _INC);

  uint32_t base[8];
  SetBigInt(base, x);
  uint32_t result[8] = {1, 0, 0, 0, 0, 0, 0, 0};

  for (int i = 255; i >= 0; i--) {
    MulModP(result, result, result);
    if ((exp[i >> 5] >> (i & 31)) & 1) {
      MulModP(result, result, base);
    }
  }

  SetBigInt(r, result);
}

__device__ void ScalarMulP(uint32_t *result_x, uint32_t *result_y,
                           const uint32_t *scalar) {
  // Pre-computed points array (2^4 = 16 points)
  uint32_t pre_x[16][8];
  uint32_t pre_y[16][8];

  // Initialize first point (generator)
  memcpy(pre_x[1], G_x, sizeof(uint32_t) * 8);
  memcpy(pre_y[1], G_y, sizeof(uint32_t) * 8);

  // Pre-compute points: 2P, 3P, ..., 15P
  for (int i = 2; i < 16; i++) {
    memcpy(pre_x[i], pre_x[i - 1], sizeof(uint32_t) * 8);
    memcpy(pre_y[i], pre_y[i - 1], sizeof(uint32_t) * 8);
    SecpAddStep(pre_x[i], pre_y[i], G_x, G_y, threadIdx.x);
  }

  // Initialize result to infinity point
  uint32_t rx[8] = {0}, ry[8] = {0};
  bool first = true;

  // Process 4 bits at a time
  for (int i = 252; i >= 0; i -= 4) {
    // Double 4 times
    if (!first) {
      for (int j = 0; j < 4; j++) {
        SecpDblStep(rx, ry, threadIdx.x);
      }
    }

    // Get 4-bit window
    int window = (scalar[i >> 5] >> (i & 31)) & 0xF;

    if (window) {
      if (first) {
        memcpy(rx, pre_x[window], sizeof(uint32_t) * 8);
        memcpy(ry, pre_y[window], sizeof(uint32_t) * 8);
        first = false;
      } else {
        SecpAddStep(rx, ry, pre_x[window], pre_y[window], threadIdx.x);
      }
    }
  }

  memcpy(result_x, rx, sizeof(uint32_t) * 8);
  memcpy(result_y, ry, sizeof(uint32_t) * 8);
}

__device__ void _ScalarMulP(uint32_t *result_x, uint32_t *result_y,
                            const uint32_t *scalar) {
  // Initialize accumulator point to infinity
  uint32_t rx[8] = {0}, ry[8] = {0};

  // Get generator point
  uint32_t px[8], py[8];
  memcpy(px, G_x, sizeof(uint32_t) * 8);
  memcpy(py, G_y, sizeof(uint32_t) * 8);

  // Double-and-add from MSB to LSB
  bool first_one = false;

  for (int i = 255; i >= 0; i--) {
    if (first_one) {
      SecpDblStep(rx, ry, threadIdx.x);
    }

    if ((scalar[i >> 5] >> (i & 31)) & 1) {
      if (!first_one) {
        memcpy(rx, px, sizeof(uint32_t) * 8);
        memcpy(ry, py, sizeof(uint32_t) * 8);
        first_one = true;
      } else {
        SecpAddStep(rx, ry, px, py, threadIdx.x);
      }
    }
  }

  memcpy(result_x, rx, sizeof(uint32_t) * 8);
  memcpy(result_y, ry, sizeof(uint32_t) * 8);
}
