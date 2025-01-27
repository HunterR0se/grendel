# File: Grendel/gpu/keygen/CMakeLists.txt

cuda_add_library(keygen SHARED
    keygen.cu
    secp256k1/brain_secp.cu
    secp256k1/constants.cu
    hash.cu
)

target_include_directories(keygen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/secp256k1
)

set_target_properties(keygen PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 89
)
