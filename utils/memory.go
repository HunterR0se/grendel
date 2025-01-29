package utils

import "runtime"

type MemStats struct {
    AllocatedGB float64
    SystemGB    float64
    TotalGB     float64
}

func GetMemStats() *MemStats {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)

    return &MemStats{
        AllocatedGB: float64(m.Alloc) / (1024 * 1024 * 1024),
        SystemGB:    float64(m.Sys) / (1024 * 1024 * 1024),
        TotalGB:     float64(m.TotalAlloc) / (1024 * 1024 * 1024),
    }
}
