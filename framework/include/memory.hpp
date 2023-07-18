#ifndef __MEMORY__
#define __MEMORY__

#include "common.hpp"

namespace geefer {

enum CpyMode {HostToDevice, DeviceToHost};

class Memory {
public:

    void GFMallocHost(void **ptr, uint32_t size);
    void GFMallocDevice(void **ptr, uint32_t size);

    inline void GFFreeHost(void* ptr) { free(ptr); }
    inline void GFFreeDevice(void* d_ptr) { 
        checkCudaErrors(cudaFree(d_ptr));
    }

    void GFMemCpy(void *host_ptr, 
                  void *device_ptr, 
                  uint32_t size,
                  CpyMode cpy_mode);

}; // class Memory

} // namespace geefer
#endif // __SYNCMEM__

