#ifndef __SYNCMEM__
#define __SYNCMEM__

#include "common.hpp"
#include "memory.hpp"

namespace geefer {

/*
inline void GFMallocHost(void **ptr, uint32_t size) {
    *ptr = malloc(size);
    if(*ptr == nullptr){
        std::string w = "pointer is null, malloc failed";
        Warning(w);
    }
}

inline void GFFreeHost(void* ptr) { free(ptr); }

inline void GFMallocDevice(void **ptr, uint32_t size) {
    //cudaMalloc(ptr, size);
    checkCudaErrors(cudaMalloc(ptr, size*sizeof(float)));
}

inline void GFMemCpy(void **ptr, uint32_t size) {
    checkCudaErrors(cudaMalloc(ptr, size*sizeof(float)));

inline void GFFreeDevice(void* ptr) { free(ptr); }
*/

class SyncMem
{
public:
    SyncMem();
    explicit SyncMem(uint32_t size, 
                     BackEnd backend=CPU);
    ~SyncMem();

    const void* cpu_data();
    const void* gpu_data();

    void set_cpu_data(void *data, uint32_t size);
    void set_gpu_data(void *data);

    void* mutable_cpu_data();
    void* mutable_gpu_data();

    /*
        UNINIT: not initialized, both cpu & gpu are not allocated
        AT_CPU: data on cpu is refreshed, but gpu not
        SYNCED: both cpu & gpu are allocated, and data are the same
     */
    enum SyncHead {UNINIT, AT_CPU, SYNCED};
    //enum CpyMode {HostToDevice, DeviceToHost};

    SyncHead head() const {return __head;}
    uint32_t cpu_size() const {return __cpu_size;}
    uint32_t gpu_size() const {return __gpu_size;}
    void sync_gpu2cpu();

private:
    void __to_cpu();
    void __to_gpu();

    void* __host_ptr;
    void* __device_ptr;
    SyncHead __head;
    uint32_t __cpu_size;
    uint32_t __gpu_size;
    bool __own_cpu_data;
    bool __own_gpu_data;
    //BackEnd __backend;

    std::shared_ptr<Memory> __mem;

    DISABLE_COPY_AND_ASSIGN(SyncMem);

}; // class SyncMem
}; // namespace geefer

#endif // __SYNCMEM__

