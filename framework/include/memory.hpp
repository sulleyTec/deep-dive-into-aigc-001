#ifndef __SYNCMEM__
#define __SYNCMEM__

#include "helper_cuda.h"
#include "common.hpp"

namespace geefer {

class Memory {
public:

    inline void GFMallocHost(void **ptr, uint32_t size);
    inline void GFFreeHost(void* ptr) { free(ptr); }
    inline void GFMallocDevice(void **ptr, uint32_t size);
    inline void GFMemCpy(void **host_ptr, void **device_ptr, 
                         uint32_t size,
                         CpyMode &cpy_mode);
    inline void GFFreeDevice(void* ptr) { free(ptr); }

protected:
    enum CpyMode {HostToDevice, DeviceToHost};

}; // class Memory

/*
class CpuMem: public Memory {
public:

private:

}; // class CpuMem

class GpuMem: public Memory {
public:


}; // class GpuMem
*/

} // namespace geefer
#endif // __SYNCMEM__

