#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include "log4cplus/logger.h"
#include "log4cplus/consoleappender.h"
#include "log4cplus/loglevel.h"
#include <log4cplus/loggingmacros.h>
#include <log4cplus/initializer.h>
#include <log4cplus/configurator.h>
#include <iomanip>

#include <string.h>
#include <math.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <utility> // pair
#include <map>
#include <set>

#include <cuda_runtime.h>
#include "helper_cuda.h"

#define CHECK_GE(x,y) ((x)>=(y)?true:false) // x is greater or equal to y
#define CHECK_GT(x,y) ((x)>=(y)?true:false) // x is greater than y
#define CHECK_LE(x,y) ((x)<=(y)?true:false) // x is less or equal y
#define CHECK_LT(x,y) ((x)<=(y)?true:false) // x is less than y

#define DISABLE_COPY_AND_ASSIGN(classname) \
private: \
    classname(const classname&); \
    classname& operator=(const classname&);

const uint32_t kMaxTensorAxes = 32;
enum BackEnd {CPU, GPU};
enum CpyMode {HostToDevice, DeviceToHost};

void Warning(const std::string &warning);

#endif

