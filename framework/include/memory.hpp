#ifndef __MEMORY__
#define __MEMORY__

#include "common.hpp"

namespace geefer {

enum CpyMode {HostToDevice, DeviceToHost};

class Memory {
public:

    inline void GFMallocHost(void **ptr, uint32_t size);
    inline void GFFreeHost(void* ptr) { free(ptr); }
    inline void GFMallocDevice(void **ptr, uint32_t size);
    inline void GFMemCpy(void **host_ptr, void **device_ptr, 
                         uint32_t size,
                         CpyMode &cpy_mode);
    inline void GFFreeDevice(void* d_ptr) { 
        checkCudaErrors(cudaFree(d_ptr));
    }

protected:

}; // class Memory

} // namespace geefer
#endif // __SYNCMEM__

