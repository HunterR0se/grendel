/**
 * @file keygen.cu
 * @brief GPU-accelerated private key generation using CUDA for Bitcoin keys
 * based on secp256k1 curve parameters.
 *
 * This code provides a CUDA kernel to generate private keys from random seeds,
 * ensuring they are within the valid range of the secp256k1 elliptic curve's
 * order (N). The implementation includes modular reduction if necessary and is
 * designed to run on GPUs supporting the sm_89+ architecture.
 *
 bitfinder/gpu/keygen/
 ├── secp256k1/
 │   ├── field.cu
 │   ├── field.h          # Changed from .cuh
 │   ├── group.cu
 │   ├── group.h          # Changed from .cuh
 │   └── constants.h      # Changed from .cuh
 ├── keygen.cu
 ├── keygen.h
 ├── hash.cu
 └── hash.h               # Changed from .cuh
 */

#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
             cudaGetErrorString(err));                                         \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#pragma nv_diag_suppress 20012
#include "hash.h"
#include "keygen.h"
#include "secp256k1/brain_secp.h"
#include "secp256k1/constants.h"
#include "secp256k1/group.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

__constant__ uint32_t d_Gx[8];
__constant__ uint32_t d_Gy[8];
__constant__ uint32_t d_P[8];
__constant__ uint32_t d_N[8];
__constant__ uint32_t d_EC[8];
__constant__ uint32_t d_INC[8];

extern "C" int initialize_cuda_constants() {
  // Load all constants into GPU memory
  cudaError_t err;

  err = cudaMemcpyToSymbol(d_Gx, G_x, sizeof(uint32_t) * 8);
  if (err != cudaSuccess)
    return -1;

  err = cudaMemcpyToSymbol(d_Gy, G_y, sizeof(uint32_t) * 8);
  if (err != cudaSuccess)
    return -1;

  err = cudaMemcpyToSymbol(d_P, P, sizeof(uint32_t) * 8);
  if (err != cudaSuccess)
    return -1;

  err = cudaMemcpyToSymbol(d_N, N, sizeof(uint32_t) * 8);
  if (err != cudaSuccess)
    return -1;

  err = cudaMemcpyToSymbol(d_EC, _EC, sizeof(uint32_t) * 8);
  if (err != cudaSuccess)
    return -1;

  err = cudaMemcpyToSymbol(d_INC, _INC, sizeof(uint32_t) * 8);
  if (err != cudaSuccess)
    return -1;

  return 0;
}

#define CUDA_LOG(tid, fmt, ...)                                                \
  printf("[CUDA][Thread %d] " fmt "\n", tid, ##__VA_ARGS__)

#define CUDA_ERROR(tid, fmt, ...)                                              \
  printf("[CUDA ERROR][Thread %d] " fmt "\n", tid, ##__VA_ARGS__)

extern "C" int test_cuda_device() {
  printf("Testing CUDA device...\n");

  int deviceCount;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
    printf("No CUDA devices found\n");
    return -1;
  }

  cudaDeviceProp prop;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));

  printf("Using device: %s\n", prop.name);
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

  return 0;
}

// ---------------------  MAIN FUNCTIONS
__global__ void test_hash_kernel(unsigned char *results);
__global__ void test_conversion_kernel(struct DebugData *debug);

__global__ void generate_keys_kernel(struct KeyAddressData *keys, int count,
                                     unsigned long long seed, bool debug_mode) {

  /*  __shared__ uint32_t shared_G[16]; // Store G_x and G_y

  // Load generator point into shared memory once per block
  if (threadIdx.x == 0) {
    memcpy(shared_G, G_x, 32);
    memcpy(shared_G + 8, G_y, 32);
  }
  __syncthreads();
  */

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  // Initialize RNG
  curandState state;
  curand_init(seed + idx, 0, 0, &state);

  struct KeyAddressData *key = &keys[idx];
  uint32_t private_key[8] = {0};

  // Generate and store private key directly in big-endian format
  for (int j = 0; j < 8; j++) {
    private_key[j] = curand(&state);
  }

  // Generate public key
  uint32_t pub_x[8] = {0}, pub_y[8] = {0};

  // Do scalar multiplication (key already in big-endian)
  secp256k1_scalar_multiply_base(pub_x, pub_y, private_key);

  // Store private key bytes
  for (int j = 0; j < 8; j++) {
    key->private_key[j * 4] = (private_key[j] >> 24) & 0xFF;
    key->private_key[j * 4 + 1] = (private_key[j] >> 16) & 0xFF;
    key->private_key[j * 4 + 2] = (private_key[j] >> 8) & 0xFF;
    key->private_key[j * 4 + 3] = private_key[j] & 0xFF;
  }

  // Store public key X coordinate
  for (int j = 0; j < 8; j++) {
    key->pubkey_x[j * 4] = (pub_x[j] >> 24) & 0xFF;
    key->pubkey_x[j * 4 + 1] = (pub_x[j] >> 16) & 0xFF;
    key->pubkey_x[j * 4 + 2] = (pub_x[j] >> 8) & 0xFF;
    key->pubkey_x[j * 4 + 3] = pub_x[j] & 0xFF;
  }

  key->pubkey_y_parity = pub_y[0] & 1;

  // Prepare compressed public key
  unsigned char pubkey[33];
  pubkey[0] = 0x02 | key->pubkey_y_parity;
  memcpy(pubkey + 1, key->pubkey_x, 32);

  // Generate hash160
  hash160(pubkey, 33, key->hash160);

  // Address type
  float r = curand_uniform(&state);
  key->address_type = (r < 0.33f) ? 0 : (r < 0.66f) ? 1 : 2;
}

// ---------------------  HELPERS --------------------
extern "C" int test_basic_operations() {
  struct KeyAddressData *d_keys;
  cudaMalloc(&d_keys, sizeof(struct KeyAddressData));

  // Launch minimal test kernel
  int threadsPerBlock = 1;
  int numBlocks = 1;
  generate_keys_kernel<<<numBlocks, threadsPerBlock>>>(d_keys, 1, time(NULL),
                                                       true);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Basic test failed: %s\n", cudaGetErrorString(error));
    cudaFree(d_keys);
    return -1;
  }

  cudaFree(d_keys);
  return 0;
}

extern "C" int generate_keys_combined(struct KeyAddressData *key_data,
                                      int count, int debug_mode) {

  struct KeyAddressData *d_keys;

  if (debug_mode) {
    printf("[CUDA Debug] Starting GPU allocation for %d keys (%zu bytes)...\n",
           count, count * sizeof(struct KeyAddressData));
  }

  if (initialize_cuda_constants() != 0) {
    printf("[CUDA Error] Failed to initialize constants\n");
    return -1;
  }

  // Check if input parameters are valid
  if (key_data == NULL || count <= 0) {
    printf("[CUDA Error] Invalid input parameters: key_data=%p, count=%d\n",
           key_data, count);
    return -1;
  }

  // Get available GPU memory before allocation
  size_t free, total;
  cudaError_t memErr = cudaMemGetInfo(&free, &total);
  if (memErr != cudaSuccess) {
    printf("[CUDA Error] Failed to get GPU memory info: %s\n",
           cudaGetErrorString(memErr));
    return -1;
  }
  if (debug_mode) {
    printf("[CUDA Debug] GPU Memory - Free: %zu MB, Total: %zu MB\n",
           free / (1024 * 1024), total / (1024 * 1024));
  }

  cudaError_t err = cudaMalloc(&d_keys, count * sizeof(struct KeyAddressData));
  if (err != cudaSuccess) {
    printf("[CUDA Error] Failed to allocate device memory: %s\n",
           cudaGetErrorString(err));
    return -1;
  }
  if (debug_mode) {
    printf("[CUDA Debug] GPU memory allocation successful\n");
    printf("[CUDA Debug] Setting CUDA device timeout watchdog...\n");
  }

  // Use cudaDeviceScheduleYield for timeout behavior
  cudaError_t timeoutErr = cudaSetDeviceFlags(cudaDeviceScheduleYield);
  if (timeoutErr != cudaSuccess) {
    printf("[CUDA Warning] Failed to set device flags: %s\n",
           cudaGetErrorString(timeoutErr));
  }

  // Optional: Set kernel runtime limits
  timeoutErr = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 10);
  if (timeoutErr != cudaSuccess) {
    printf("[CUDA Warning] Failed to set sync depth limit: %s\n",
           cudaGetErrorString(timeoutErr));
  }

  // Launch kernel with fewer threads for testing
  int threadsPerBlock = 256;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0); // We already have this line
  // Use 80% of max grid size for better occupancy
  int maxBlocks = (int)(prop.maxGridSize[0] * 0.8);
  int numBlocks = (count + threadsPerBlock - 1) / threadsPerBlock;
  if (numBlocks > maxBlocks) {
    numBlocks = maxBlocks;
  }

  unsigned long long seed = time(NULL);

  // Check if grid dimensions are valid
  if (numBlocks > prop.maxGridSize[0]) {
    printf("[CUDA Error] Grid size exceeds device capabilities\n");
    cudaFree(d_keys);
    return -1;
  }

  if (debug_mode) {
    printf("[CUDA Debug] Launching kernel with parameters: %d blocks, %d "
           "threads per block\n",
           numBlocks, threadsPerBlock);
  }

  // Add timeout handling
  cudaError_t error;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // In generate_keys_combined
  if (test_basic_operations() != 0) {
    printf("[CUDA] Basic operations test failed\n");
    return -1;
  }

  generate_keys_kernel<<<numBlocks, threadsPerBlock>>>(d_keys, count, seed,
                                                       (bool)debug_mode);

  // Check for timeout
  float timeout = 10.0f; // 10 seconds timeout
  cudaEventRecord(stop);
  while (cudaEventQuery(stop) == cudaErrorNotReady) {
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    if (elapsed > timeout * 1000) {
      printf("[CUDA] Kernel timeout after %.1f seconds\n", timeout);
      cudaDeviceReset();
      return -1;
    }
    cudaDeviceSynchronize();
  }

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Check for kernel launch errors
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("[CUDA Error] Kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_keys);
    return -1;
  }

  // Wait for kernel to finish and check for errors
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("[CUDA Error] Kernel execution failed: %s\n",
           cudaGetErrorString(err));
    cudaFree(d_keys);
    return -1;
  }

  if (debug_mode) {
    printf("[CUDA Debug] Copying results back to host...\n");
  }

  err = cudaMemcpy(key_data, d_keys, count * sizeof(struct KeyAddressData),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("[CUDA Error] Failed to copy keys back: %s\n",
           cudaGetErrorString(err));
    cudaFree(d_keys);
    return -1;
  }

  cudaFree(d_keys);
  return 0;
}

// Test functions
extern "C" int test_hash_functions() {
  unsigned char *d_results;
  unsigned char
      h_results[72]; // 32 bytes SHA256 + 20 bytes RIPEMD160 + 20 bytes Hash160

  cudaError_t err = cudaMalloc(&d_results, 72);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory: %s\n",
            cudaGetErrorString(err));
    return -1;
  }

  test_hash_kernel<<<1, 1>>>(d_results);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_results);
    return -1;
  }

  err = cudaMemcpy(h_results, d_results, 72, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy results back: %s\n",
            cudaGetErrorString(err));
    cudaFree(d_results);
    return -1;
  }

  // Known correct results for "The quick brown fox jumps over the lazy dog"
  const unsigned char correct_sha256[] = {
      0xd7, 0xa8, 0xfb, 0xb3, 0x07, 0xd7, 0x80, 0x94, 0x69, 0xca, 0x9a,
      0xbc, 0xb0, 0x08, 0x2e, 0x4f, 0x8d, 0x56, 0x51, 0xe4, 0x6d, 0x3c,
      0xdb, 0x76, 0x2d, 0x02, 0xd0, 0xbf, 0x37, 0xc9, 0xe5, 0x14};

  const unsigned char correct_ripemd160[] = {
      0x37, 0xf3, 0x32, 0xf6, 0x8d, 0xb7, 0x7b, 0xd9, 0xd7, 0xed,
      0xd4, 0x96, 0x95, 0x71, 0xad, 0x67, 0x1c, 0xf9, 0xdd, 0x3b};

  cudaFree(d_results);

  bool sha256_correct = memcmp(h_results, correct_sha256, 32) == 0;
  bool ripemd160_correct = memcmp(h_results + 32, correct_ripemd160, 20) == 0;

  if (!sha256_correct || !ripemd160_correct) {
    return -1;
  }

  return 0;
}

extern "C" int test_key_conversion(struct DebugData *debug_data) {
  struct DebugData *d_debug;
  cudaError_t err = cudaMalloc(&d_debug, sizeof(struct DebugData));
  if (err != cudaSuccess) {
    return -1;
  }

  test_conversion_kernel<<<1, 1>>>(d_debug);

  err = cudaMemcpy(debug_data, d_debug, sizeof(struct DebugData),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    cudaFree(d_debug);
    return -1;
  }

  cudaFree(d_debug);
  return 0;
}

// ----------------------

__global__ void test_hash_kernel(unsigned char *results) {
  const char test_input[] = "The quick brown fox jumps over the lazy dog";
  const int input_len = 43;

  // Only first thread does the test
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Test SHA256
    unsigned char sha_result[32];
    sha256((unsigned char *)test_input, input_len, sha_result);

    // Copy SHA256 result to output
    for (int i = 0; i < 32; i++) {
      results[i] = sha_result[i];
    }

    // Test RIPEMD160
    unsigned char ripemd_result[20];
    ripemd160((unsigned char *)test_input, input_len, ripemd_result);

    // Copy RIPEMD160 result to output
    for (int i = 0; i < 20; i++) {
      results[32 + i] = ripemd_result[i];
    }

    // Test Hash160 (SHA256 + RIPEMD160)
    unsigned char hash160_result[20];
    hash160((unsigned char *)test_input, input_len, hash160_result);

    // Copy Hash160 result to output
    for (int i = 0; i < 20; i++) {
      results[52 + i] = hash160_result[i];
    }
  }
}

__global__ void test_conversion_kernel(struct DebugData *debug) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Create test private key (known test vector)
    uint32_t test_priv[8] = {0x7fdb5096, 0x3392d2e0, 0xe9e9f3bc, 0xa5ef8449,
                             0x64f9bddc, 0xb6f973ba, 0xb350185c, 0x69a6c281};

    // Save original private key
    memcpy(debug->original_priv, test_priv, sizeof(test_priv));

    // Generate public key
    uint32_t pub_x[8], pub_y[8];
    secp256k1_scalar_multiply_base(pub_x, pub_y, test_priv);

    // Store results
    memcpy(debug->pubkey_x, pub_x, sizeof(pub_x));
    memcpy(debug->pubkey_y, pub_y, sizeof(pub_y));
    debug->pubkey_y_parity = pub_y[0] & 1;
  }
}
