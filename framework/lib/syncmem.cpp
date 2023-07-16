#include "syncmem.hpp"

namespace geefer {

SyncMem::SyncMem(): 
    __cpu_ptr(nullptr), __size(0ul), 
    __head(UNINIT), __own_cpu_data(false) {}

SyncMem::SyncMem(uint32_t size): 
    __cpu_ptr(nullptr), __size(size), 
    __head(UNINIT), __own_cpu_data(false) {}

SyncMem::~SyncMem() {
    if(__cpu_ptr) {
        GFFreeHost(__cpu_ptr);
    }
}

void SyncMem::to_cpu() {
    switch(__head) {
        case UNINIT:
            GFMallocHost(&__cpu_ptr, __size);
            memset(__cpu_ptr, 0, __size);
            __own_cpu_data = true;
            break;
        case AT_CPU:
            break;
        case AT_GPU:
            break;
        case SYNCED:
            break;
    }
}

const void* SyncMem::cpu_data() {
    to_cpu();
    return const_cast<const void*>(__cpu_ptr);
}

void SyncMem::set_cpu_data(void *data, uint32_t size) {
    if(data == nullptr) {
        std::string warn = "data pointer is null";
        Warning(warn);
        return; 
    }

    if(__own_cpu_data) {
        GFFreeHost(__cpu_ptr);
    }

    /* deep copy for safe */
    __size = size;
    GFMallocHost(&__cpu_ptr, __size);
    memcpy(__cpu_ptr, data, __size);
    __own_cpu_data = true;
    __head = AT_CPU;
}

void* SyncMem::mutable_cpu_data() {
    to_cpu();
    __head = AT_CPU;
    return __cpu_ptr;
}

}; //namespace geefer

