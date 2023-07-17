#include "memory.hpp"

namespace geefer {

inline void Memory::GFMallocHost(void **ptr, uint32_t size) {
    *ptr = malloc(size);
    if(*ptr == nullptr){
        std::string w = "pointer is null, malloc failed";
        Warning(w);
    }
}

inline void Memory::GFMallocDevice(void **ptr, uint32_t size) {
    checkCudaErrors(cudaMalloc(ptr, size*sizeof(float)));
}

inline void Memory::GFMemCpy(void **host_ptr, 
                             void **device_ptr,
                             uint32_t size, 
                             CpyMode &cpy_mode) {
    switch(cpy_mode) {
        case HostToDevice:
            checkCudaErrors(cudaMemcpy(device_ptr, host_ptr, 
                                       size*sizeof(float), 
                                       cudaMemcpyHostToDevice));
            break;
        case DeviceToHost:
            checkCudaErrors(cudaMemcpy(host_ptr, device_ptr, 
                                       size*sizeof(float), 
                                       cudaMemcpyDeviceToHost));
            break;
    }
}

} //namespace geefer


