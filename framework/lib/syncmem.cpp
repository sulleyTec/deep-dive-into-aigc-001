#include "syncmem.hpp"

namespace geefer {

SyncMem::SyncMem(): 
    __host_ptr(nullptr), __cpu_size(0ul), __gpu_size(0ul),
    __head(UNINIT), __own_cpu_data(false),
    __own_gpu_data(false), __mem(std::make_shared<Memory>()) {}

SyncMem::SyncMem(uint32_t size, BackEnd backend): 
    __host_ptr(nullptr), __cpu_size(size), __gpu_size(size),
    __head(UNINIT), __own_cpu_data(false),
    __own_gpu_data(false) {}

SyncMem::~SyncMem() {
    if(__host_ptr) {
        __mem->GFFreeHost(__host_ptr);
    }
    if(__device_ptr){
        __mem->GFFreeDevice(__device_ptr);
    }
}

void SyncMem::__to_cpu() {
    switch(__head) {
        case UNINIT:
            /* only malloc cpu data */
            __mem->GFMallocHost(&__host_ptr, __cpu_size);
            memset(__host_ptr, 0, __cpu_size);
            __own_cpu_data = true;
            __head = AT_CPU;
            break;
        case AT_CPU:
            break;
        case SYNCED:
            break;
    }
}

void SyncMem::__to_gpu() {
    switch(__head) {
        case UNINIT:
            __mem->GFMallocHost(&__host_ptr, __cpu_size);
            memset(__host_ptr, 0, __cpu_size);
            __own_cpu_data = true;

            /* malloc gpu data */
            __mem->GFMallocDevice(&__device_ptr, __gpu_size);
            /* sync cpu data to gpu */
            if(__cpu_size!=__gpu_size) {
                Warning("size of gpu and cpu does not equal");
                exit(1);
            }
            __mem->GFMemCpy(&__host_ptr, 
                            &__device_ptr, 
                            __gpu_size,
                            HostToDevice);
            __own_gpu_data = true;
            __head = SYNCED;
            break;
        case AT_CPU:
            if(!__own_gpu_data || __cpu_size!=__gpu_size) {
                if (__cpu_size!=__gpu_size) {
                    __mem->GFFreeDevice(__device_ptr);
                    __gpu_size = __cpu_size;
                }

                /* malloc gpu data */
                __mem->GFMallocDevice(&__device_ptr, __cpu_size);
            }

            /* sync cpu data to gpu */
            __mem->GFMemCpy(&__host_ptr, 
                            &__device_ptr, 
                            __gpu_size,
                            HostToDevice);

            __own_gpu_data = true;
            __head = SYNCED;
            break;
        case SYNCED:
            break;
    }
}

const void* SyncMem::cpu_data() {
    __to_cpu();
    return const_cast<const void*>(__host_ptr);
}

const void* SyncMem::gpu_data() {
    __to_gpu();
    return const_cast<const void*>(__device_ptr);
}

void SyncMem::set_gpu_data(void *data, uint32_t size) {
    set_cpu_data(data, size);
    __to_gpu();
}

void SyncMem::set_cpu_data(void *data, uint32_t size) {
    if(data == nullptr) {
        std::string warn = "data pointer is null";
        Warning(warn);
        return; 
    }

    if(__own_cpu_data) {
        __mem->GFFreeHost(__host_ptr);
    }

    /* deep copy for safe */
    __size = size;
    __mem->GFMallocHost(&__host_ptr, __size);
    memcpy(__host_ptr, data, __size);
    __own_cpu_data = true;
    __head = AT_CPU;
}

void* SyncMem::mutable_cpu_data() {
    __to_cpu();
    return __host_ptr;
}

void* SyncMem::mutable_gpu_data() {
    __to_gpu();
    return __device_ptr;
}

void sync_gpu2cpu() {
    if(__gpu_size!=__cpu_size) {
        Warning("size of gpu and cpu do not match");
        exit(1);
    }

    __mem->GFMemCpy(&__host_ptr, 
                    &__device_ptr, 
                    __gpu_size,
                    DeviceToHost);
    __head = SYNCED;
}

}; //namespace geefer

