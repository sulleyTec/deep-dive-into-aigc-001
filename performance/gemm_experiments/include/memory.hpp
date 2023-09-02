#ifndef __MEM_HPP__
#define __MEM_HPP__

#include "common.hpp"

class BaseMemory {
public:
    BaseMemory(): cpu_(nullptr), cpu_bytes_(0ul), cpu_capacity_(0ul), owner_cpu_(true),
                  gpu_(nullptr), gpu_bytes_(0ul), gpu_capacity_(0ul), owner_gpu_(true) {}

    BaseMemory(void *cpu, size_t cpu_bytes, 
               void *gpu, size_t gpu_bytes);
    virtual ~BaseMemory();

    virtual void *gpu_realloc(size_t bytes);
    virtual void *cpu_realloc(size_t bytes);
    void release_gpu();
    void release_cpu();
    void release();
    inline size_t cpu_bytes() const { return cpu_bytes_; }
    inline size_t gpu_bytes() const { return gpu_bytes_; }
    virtual inline void *get_gpu() const { return gpu_; }
    virtual inline void *get_cpu() const { return cpu_; }
    void reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);

protected:
    void *cpu_;
    uint32_t cpu_bytes_;
    uint32_t cpu_capacity_;
    bool owner_cpu_;

    void *gpu_;
    uint32_t gpu_bytes_;
    uint32_t gpu_capacity_;
    bool owner_gpu_;
};

template <typename DT>
class Memory: public BaseMemory {
public:
    Memory() = default;

    /* avoid shallow copy */
    Memory(const Memory &other) = delete;
    Memory &operator=(const Memory &other) = delete;

    virtual DT *gpu(size_t size) { 
        return (DT *)BaseMemory::gpu_realloc(size * sizeof(DT)); 
    }

    virtual DT *cpu(size_t size) { 
        return (DT *)BaseMemory::cpu_realloc(size * sizeof(DT)); 
    }

    inline size_t cpu_size() const { return cpu_bytes_ / sizeof(DT); }
    inline size_t gpu_size() const { return gpu_bytes_ / sizeof(DT); }

    virtual inline DT *gpu() const { return (DT *)gpu_; }
    virtual inline DT *cpu() const { return (DT *)cpu_; }
};

#endif
