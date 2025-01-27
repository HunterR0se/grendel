package gpu

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	cuda "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

// GPUInfo contains basic information about the detected GPU
type GPUInfo struct {
	Available bool   // Whether a GPU is available
	Name      string // GPU name/description
	Model     string // GPU model/vendor
	VRAM      int64  // VRAM in GB (if available)
	UsingCUDA bool   // Whether CUDA is being used
}

// CheckForGPU returns information about available GPU

func CheckForGPU() GPUInfo {
	info := GPUInfo{}

	// Check for CUDA support first
	cudaAvailable := false
	if count, err := cuda.GetDeviceCount(); err == cuda.CudaSuccess && count > 0 {
		cudaAvailable = true
	}

	// Try NVIDIA first using nvidia-smi
	if output, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader").Output(); err == nil {
		parts := strings.Split(strings.TrimSpace(string(output)), ", ")
		if len(parts) >= 2 {
			if vram, err := parseVRAMSize(parts[1], "MiB", 1024); err == nil {
				return GPUInfo{
					Available: true,
					Name:      parts[0],
					Model:     "NVIDIA",
					VRAM:      vram,
					UsingCUDA: cudaAvailable,
				}
			}
		}
	}

	// Fallback to basic lspci check
	if output, err := exec.Command("lspci", "-nn").Output(); err == nil {
		for _, line := range strings.Split(string(output), "\n") {
			if strings.Contains(line, "[10de:") { // NVIDIA vendor ID
				return GPUInfo{Available: true, Name: line, Model: "NVIDIA", VRAM: 8}
			}
			if strings.Contains(line, "[1002:") { // AMD vendor ID
				return GPUInfo{Available: true, Name: line, Model: "AMD", VRAM: 8}
			}
		}
	}

	info.UsingCUDA = cudaAvailable
	return info
}

func parseVRAMSize(sizeStr string, unit string, conversionFactor int) (int64, error) {
	parts := strings.SplitN(sizeStr, " ", 2)
	if len(parts) < 1 {
		return 0, fmt.Errorf("invalid size format")
	}
	value, err := strconv.ParseInt(parts[0], 10, 64)
	if err != nil {
		return 0, err
	}
	if strings.HasSuffix(sizeStr, unit) || (len(parts) == 2 && strings.TrimSpace(parts[1]) == unit) {
		return value / int64(conversionFactor), nil
	}
	return 0, fmt.Errorf("unexpected unit in size: %s", unit)
}
