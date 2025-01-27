#ifndef KEYGEN_H
#define KEYGEN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

struct KeyAddressData {
  unsigned char private_key[32];
  unsigned char pubkey_x[32];
  unsigned char pubkey_y_parity;
  unsigned char hash160[20];
  unsigned char address_type;
};

struct DebugData {
  uint32_t original_priv[8];
  uint32_t pubkey_x[8];
  uint32_t pubkey_y[8];
  uint8_t pubkey_y_parity;
};

// Add this new function declaration
extern void set_cuda_debug_mode(int enabled);

// Existing function declarations
// Update the declaration to match
extern int generate_keys_combined(struct KeyAddressData *key_data, int count,
                                  int debug_mode);

extern int test_hash_functions(void);
extern int test_key_conversion(struct DebugData *debug);
extern int test_cuda_device(void);

#ifdef __cplusplus
}
#endif

#endif // KEYGEN_H
