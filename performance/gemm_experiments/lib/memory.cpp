#include "memory.hpp"
#include "log.hpp"

BaseMemory::BaseMemory(void *cpu, 
                       size_t cpu_bytes, 
                       void *gpu, 
                       size_t gpu_bytes) {

  reference(cpu, cpu_bytes, gpu, gpu_bytes);
}

BaseMemory::~BaseMemory() { 
    release(); 
}

void BaseMemory::reference(void *cpu, 
                           size_t cpu_bytes, 
                           void *gpu, 
                           size_t gpu_bytes) {
    release();

    if (cpu == nullptr || cpu_bytes == 0) {
        cpu = nullptr;
        cpu_bytes = 0;
    }

    if (gpu == nullptr || gpu_bytes == 0) {
        gpu = nullptr;
        gpu_bytes = 0;
    }

    cpu_ = cpu;
    gpu_ = gpu;

    cpu_bytes_ = cpu_bytes;
    cpu_capacity_ = cpu_bytes;

    gpu_bytes_ = gpu_bytes;
    gpu_capacity_ = gpu_bytes;

    owner_cpu_ = !(cpu && cpu_bytes > 0);
    owner_gpu_ = !(gpu && gpu_bytes > 0);
}

void *BaseMemory::gpu_realloc(size_t bytes) {
    if (gpu_capacity_ < bytes) {
        release_gpu();

        gpu_capacity_ = bytes;
        checkRuntime(cudaMalloc(&gpu_, bytes));
    }
    gpu_bytes_ = bytes;
    return gpu_;
}

void *BaseMemory::cpu_realloc(size_t bytes) {
    if (cpu_capacity_ < bytes) {
        release_cpu();

        cpu_capacity_ = bytes;
        checkRuntime(cudaMallocHost(&cpu_, bytes));
        Assert(cpu_ != nullptr);
    }

    cpu_bytes_ = bytes;
    return cpu_;
}
// (void *) 0x7fff49c00200; cpu_capacity_=526336
void BaseMemory::release_cpu() {
    if(cpu_) {
        if(owner_cpu_) {
            checkRuntime(cudaFreeHost(cpu_));
        }
        cpu_ = nullptr;
    }

    cpu_capacity_=0;
    cpu_bytes_=0;
}

void BaseMemory::release_gpu() {
    if (gpu_) {
        if (owner_gpu_) {
            checkRuntime(cudaFree(gpu_));
        }
        gpu_ = nullptr;
    }

    gpu_capacity_ = 0;
    gpu_bytes_ = 0;
}

void BaseMemory::release() {
    release_cpu();
    release_gpu();
}

