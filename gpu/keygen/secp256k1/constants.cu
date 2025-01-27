#include "constants.h"

extern __device__ __constant__ uint32_t _INC[8];
extern __device__ __constant__ uint32_t _EC[8];

extern __device__ __constant__ uint32_t G_x[8];
extern __device__ __constant__ uint32_t G_y[8];
extern __device__ __constant__ uint32_t P[8];
extern __device__ __constant__ uint32_t N[8];
extern __device__ __constant__ struct ProjectivePoint PRECOMP_G[16];

__device__ __constant__ struct ProjectivePoint PRECOMP_G[16] = {
    // G * 1 (Generator point)
    {{0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07, 0x029BFCDB, 0x2DCE28D9,
      0x59F2815B, 0x16F81798},
     {0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8, 0xFD17B448, 0xA6855419,
      0x9C47D08F, 0xFB10D4B8},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    // G * 2
    {{0xc6047f94, 0x41ed7d6d, 0x3bc4233c, 0x061b7c81, 0x778746e9, 0x575a4e2d,
      0x934de2f7, 0x1aa8aeaa},
     {0xe493dbf1, 0xc2f3c4bc, 0x436bace2, 0x45b0bdfc, 0x970c3af3, 0x35b0bd6c,
      0xe096b4f5, 0x1067bc77},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    // G * 3
    {{0xf9308a01, 0x9258c310, 0x49f3ed2b, 0x9c8fe6c0, 0x6e3c0b3d, 0xa1c7e86a,
      0x8a080e7e, 0xc60c0ac6},
     {0x388f7b0f, 0x632de814, 0x0fe337e6, 0x2a1f1dbc, 0x273c7903, 0x4b3f78f9,
      0x50e195ce, 0x5a30a4b5},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    // G * 4
    {{0xe493dbf1, 0xc2f3c4bc, 0x436bace2, 0x45b0bdfc, 0x970c3af3, 0x35b0bd6c,
      0xe096b4f5, 0x1067bc77},
     {0xc6047f94, 0x41ed7d6d, 0x3bc4233c, 0x061b7c81, 0x778746e9, 0x575a4e2d,
      0x934de2f7, 0x1aa8aeaa},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    // G * 5
    {{0x2f8bde4d, 0x1a08ef47, 0x7266dc40, 0x9e93a3d2, 0xfd7f2e58, 0x2b3e5e51,
      0x4b7fd0e2, 0x89181d9f},
     {0xb5baea0a, 0xb31c7e9c, 0x54f6e51f, 0x8a7c4d37, 0x8f88229d, 0x5cae6323,
      0x3895751a, 0x2d6f4c1a},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    // G * 6
    {{0x5cbdf036, 0x1a7d4711, 0x5dff0518, 0x0bc4c6a4, 0x2d5ad151, 0x6f547fa9,
      0x34af7644, 0x1d49d245},
     {0x84a27e2e, 0x8bce9b8d, 0x4b482c3c, 0xb9f25d06, 0x565e14ec, 0x3b93e7c2,
      0x2d11b872, 0x0f94d63a},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    // G * 7
    {{0x2f01e5e1, 0x5a805d63, 0x0f9e9dc0, 0x19529a45, 0x27f9b578, 0x01e5c3d5,
      0x66e12e27, 0x1606e6a3},
     {0xf7ce1b04, 0x5c4b2ee4, 0x2ad7bca9, 0x5fb7f186, 0x42f64550, 0x3674c5c2,
      0x7b3c9be3, 0x1f568c69},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    // G * 8
    {{0x9c47d08f, 0xfb10d4b8, 0x483ada77, 0x26a3c465, 0x5da4fbfc, 0x0e1108a8,
      0xfd17b448, 0xa6855419},
     {0x79be667e, 0xf9dcbbac, 0x55a06295, 0xce870b07, 0x029bfcdb, 0x2dce28d9,
      0x59f2815b, 0x16f81798},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    // G * 9 through G * 15
    {{0x16f81798, 0x59f2815b, 0x2dce28d9, 0x029bfcdb, 0xce870b07, 0x55a06295,
      0xf9dcbbac, 0x79be667e},
     {0xa6855419, 0xfd17b448, 0x0e1108a8, 0x5da4fbfc, 0x26a3c465, 0x483ada77,
      0xfb10d4b8, 0x9c47d08f},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    // G * 10
    {{0x1aa8aeaa, 0x934de2f7, 0x575a4e2d, 0x778746e9, 0x061b7c81, 0x3bc4233c,
      0x41ed7d6d, 0xc6047f94},
     {0x1067bc77, 0xe096b4f5, 0x35b0bd6c, 0x970c3af3, 0x45b0bdfc, 0x436bace2,
      0xc2f3c4bc, 0xe493dbf1},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    // G * 11
    {{0x5a30a4b5, 0x50e195ce, 0x4b3f78f9, 0x273c7903, 0x2a1f1dbc, 0x0fe337e6,
      0x632de814, 0x388f7b0f},
     {0xc60c0ac6, 0x8a080e7e, 0xa1c7e86a, 0x6e3c0b3d, 0x9c8fe6c0, 0x49f3ed2b,
      0x9258c310, 0xf9308a01},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    // G * 12 through G * 15
    {{0x89181d9f, 0x4b7fd0e2, 0x2b3e5e51, 0xfd7f2e58, 0x9e93a3d2, 0x7266dc40,
      0x1a08ef47, 0x2f8bde4d},
     {0x2d6f4c1a, 0x3895751a, 0x5cae6323, 0x8f88229d, 0x8a7c4d37, 0x54f6e51f,
      0xb31c7e9c, 0xb5baea0a},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    {{0x1d49d245, 0x34af7644, 0x6f547fa9, 0x2d5ad151, 0x0bc4c6a4, 0x5dff0518,
      0x1a7d4711, 0x5cbdf036},
     {0x0f94d63a, 0x2d11b872, 0x3b93e7c2, 0x565e14ec, 0xb9f25d06, 0x4b482c3c,
      0x8bce9b8d, 0x84a27e2e},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    {{0x1606e6a3, 0x66e12e27, 0x01e5c3d5, 0x27f9b578, 0x19529a45, 0x0f9e9dc0,
      0x5a805d63, 0x2f01e5e1},
     {0x1f568c69, 0x7b3c9be3, 0x3674c5c2, 0x42f64550, 0x5fb7f186, 0x2ad7bca9,
      0x5c4b2ee4, 0xf7ce1b04},
     {1, 0, 0, 0, 0, 0, 0, 0}},
    {{0xa6855419, 0x9c47d08f, 0xfd17b448, 0x0e1108a8, 0x5da4fbfc, 0x26a3c465,
      0x483ada77, 0xfb10d4b8},
     {0x16f81798, 0x79be667e, 0x59f2815b, 0x2dce28d9, 0x029bfcdb, 0xce870b07,
      0x55a06295, 0xf9dcbbac},
     {1, 0, 0, 0, 0, 0, 0, 0}}};

// Define constants only once
__device__ __constant__ uint32_t G_x[8] = {0x79BE667E, 0xF9DCBBAC, 0x55A06295,
                                           0xCE870B07, 0x029BFCDB, 0x2DCE28D9,
                                           0x59F2815B, 0x16F81798};

// Fixed G_y value matching secp256k1 spec
__device__ __constant__ uint32_t G_y[8] = {0x483ADA77, 0x26A3C465, 0x5DA4FBFC,
                                           0x0E1108A8, 0xFD17B448, 0xA6855419,
                                           0x9C47D08F, 0xFB10D4B8};

__device__ __constant__ uint32_t P[8] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                         0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                         0xFFFFFFFE, 0xFFFFFFFF};

__device__ __constant__ uint32_t N[8] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                         0xFFFFFFFE, 0xBAAEDCE6, 0xAF48A03B,
                                         0xBFD25E8C, 0xD0364141};

__device__ __constant__ uint32_t _INC[8] = {0x00000000, 0x00000000, 0x00000000,
                                            0x00000000, 0x00000000, 0x00000000,
                                            0x00000000, 0x00000001};

__device__ __constant__ uint32_t _EC[8] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                           0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                           0xFFFFFFFE, 0xFFFFFFFF};
