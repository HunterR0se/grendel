CUDA_PATH ?= /usr/local/cuda
CUDA_INC_PATH ?= $(CUDA_PATH)/include
CUDA_BIN_PATH ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH ?= $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# Optimize for maximum performance
CFLAGS = -std=c++11 -O3 -use_fast_math
CUDA_CFLAGS = -I$(CUDA_INC_PATH) \
              -arch=sm_89 \
              -O3 \
              --use_fast_math \
              --maxrregcount=64 \
              -lineinfo \
              -I. \
              --ptxas-options=-v \
              -Xcompiler -fPIC \
              -Xptxas -O3,-v

# Enable L1 cache for global memory
CUDA_CFLAGS += -Xptxas -dlcm=ca

# Maximize register usage before spilling to local memory
CUDA_CFLAGS += -maxrregcount=255

CUDA_LDFLAGS = -L$(CUDA_LIB_PATH) -lcudart -lcurand

# Add device linking flag
NVCC_FLAGS = -dc

OBJECTS = secp256k1/constants.o \
          secp256k1/brain_secp.o \
          secp256k1/group.o \
          hash.o \
          keygen.o

TARGET = libkeygen.so

all: $(TARGET)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) $(CUDA_CFLAGS) -Xcompiler -fPIC -c $< -o $@

dlink.o: $(OBJECTS)
	$(NVCC) $(CFLAGS) $(CUDA_CFLAGS) -Xcompiler -fPIC -dlink $(OBJECTS) -o $@

$(TARGET): $(OBJECTS) dlink.o
	$(NVCC) -shared $(OBJECTS) dlink.o -o $@ $(CUDA_LDFLAGS)

clean:
	rm -f $(OBJECTS) dlink.o $(TARGET)

.PHONY: all clean
