#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <cuda_fp16.h>

#include "elementwise.hpp"

using namespace std;

//#define NUM_ELEMENTS 32 * 1024 * 1024
//#define NUM_ELEMENTS 163840*1000
#define BAND_WIDTH 912

// elementwise implementation copyed from https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/elementwise.cuh
constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;

inline cudaError_t GetNumBlocks(int64_t n, int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev); // dev will be filled with gpu id after cudaGetDevice
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                   sm_count * tpm / kBlockSize * kNumWaves));
  return cudaSuccess;
}

template<typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

template<typename T, int pack_size>
union Pack {
  static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "");
  __device__ Pack() { // ctr
    // do nothing
  }
  PackType<T, pack_size> storage;
  T elem[pack_size];
};

template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed {
  __device__ Packed() {
    // do nothing
  }
  union {
    T elem[pack_size];
  };
};

constexpr int kMaxPackBytes = 128 / 8;
constexpr int kMaxPackSize = 8;

constexpr int Min(int a, int b) { return a < b ? a : b; }

template<typename T>
constexpr int PackSize() {
  return Min(kMaxPackBytes / sizeof(T), kMaxPackSize);
}

template<typename T, typename U, typename... Args>
constexpr int PackSize() {
  return Min(PackSize<T>(), PackSize<U, Args...>());
}

template<typename T>
class HasApply2 {
  typedef char one;
  struct two {
    char x[2];
  };

  template<typename C>
  static one test(decltype(&C::Apply2));
  template<typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

template<int pack_size, typename FunctorT, typename R, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == true && pack_size % 2 == 0,
                                   Packed<R, pack_size>>::type
ApplyPack(const FunctorT& functor, const Packed<IN, pack_size>... in) {
  Packed<R, pack_size> ret;
//#pragma unroll
  for (int j = 0; j < pack_size; j += 2) { functor.Apply2(ret.elem + j, (in.elem + j)...); }
  return ret;
}

template<int pack_size, typename FunctorT, typename R, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == false || pack_size % 2 != 0,
                                   Packed<R, pack_size>>::type
ApplyPack(const FunctorT& functor, const Packed<IN, pack_size>... in) {
  Packed<R, pack_size> ret;
//#pragma unroll
  for (int j = 0; j < pack_size; ++j) { ret.elem[j] = functor((in.elem[j])...); }
  return ret;
}

template<int pack_size, typename FactoryT, typename R, typename... IN>
__global__ void __launch_bounds__(kBlockSize)
    ApplyGeneric(FactoryT factory, int64_t n_pack, Packed<R, pack_size>* pack_r,
                 const Packed<IN, pack_size>*... pack_in, int64_t n_tail, R* tail_r,
                 const IN*... tail_in) {
  auto functor = factory();
  const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
  for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x) {
    pack_r[i] = ApplyPack<pack_size, decltype(functor), R, IN...>(functor, (pack_in[i])...);
  }
  if (global_tid < n_tail) { tail_r[global_tid] = functor((tail_in[global_tid])...); }
}

template<typename FunctorT>
struct SimpleFactory {
  explicit SimpleFactory(FunctorT functor) : tpl(functor) {}
  __device__ FunctorT operator()() const { return tpl; }

 private:
  FunctorT tpl;
};

template<size_t pack_size>
bool IsAlignedForPack() {
  return true;
}

template<size_t pack_size, typename T, typename... Args>
bool IsAlignedForPack(const T* ptr, const Args*... others) {
  // uintptr_t => unsigned long
  return reinterpret_cast<uintptr_t>(ptr) % sizeof(Pack<T, pack_size>) == 0
         && IsAlignedForPack<pack_size, Args...>(others...);
}

template<size_t pack_size, typename FactoryT, typename R, typename... IN>
cudaError_t LaunchKernel(FactoryT factory, int* bd, int64_t n, R* r, const IN*... in) {
  const int64_t n_pack = n / pack_size;
  const int64_t tail_offset = n_pack * pack_size;
  const int64_t n_tail = n - tail_offset;
  int num_blocks;
  {
    cudaError_t err = GetNumBlocks(n_pack, &num_blocks);
    if (err != cudaSuccess) { return err; }
    *bd = num_blocks;
  }
  ApplyGeneric<pack_size, FactoryT, R, IN...><<<num_blocks, kBlockSize, 0>>>(
      factory, n_pack, reinterpret_cast<Packed<R, pack_size>*>(r),
      (reinterpret_cast<const Packed<IN, pack_size>*>(in))..., n_tail, r + tail_offset,
      (in + tail_offset)...);
  return cudaPeekAtLastError();
}

template<typename FactoryT, typename R, typename... IN>
struct GenericLauncher {
  static cudaError_t Launch(FactoryT factory, int* bd, int64_t n, R* r, const IN*... in) {
    constexpr int max_pack_size = PackSize<R, IN...>();
    if (IsAlignedForPack<max_pack_size, R, IN...>(r, in...)) {
      return LaunchKernel<max_pack_size, FactoryT, R, IN...>(factory, bd, n, r, in...);
    } else {
      return LaunchKernel<1, FactoryT, R, IN...>(factory, bd, n, r, in...);
    }
  }
};

template<typename FactoryT, typename R, typename A>
inline cudaError_t UnaryWithFactory(FactoryT factory, int* bd, int64_t n, R* r, const A* a) {
  return GenericLauncher<FactoryT, R, A>::Launch(factory, bd, n, r, a);
}

template<typename FunctorT, typename R, typename A>
inline cudaError_t Unary(FunctorT functor, int* bd, int64_t n, R* r, const A* a) {
  return UnaryWithFactory(SimpleFactory<FunctorT>(functor), bd, n, r, a);
}

template<typename FactoryT, typename R, typename A, typename B>
inline cudaError_t BinaryWithFactory(FactoryT factory, int* bd, int64_t n, R* r, const A* a, const B* b) {
  return GenericLauncher<FactoryT, R, A, B>::Launch(factory, bd, n, r, a, b);
}

template<typename FunctorT, typename R, typename A, typename B>
inline cudaError_t Binary(FunctorT functor, int* bd, int64_t n, R* r, const A* a, const B* b) {
  return BinaryWithFactory(SimpleFactory<FunctorT>(functor), bd, n, r, a, b);
}

template<typename FactoryT, typename R, typename A, typename B, typename C>
inline cudaError_t TernaryWithFactory(FactoryT factory, int* bd, int64_t n, R* r, const A* a, const B* b,
                                      const C* c) {
  return GenericLauncher<FactoryT, R, A, B, C>::Launch(factory, bd, n, r, a, b, c);
}

template<typename FunctorT, typename R, typename A, typename B, typename C>
inline cudaError_t Ternary(FunctorT functor, int64_t n, R* r, const A* a, const B* b, const C* c) {
  return TernaryWithFactory(SimpleFactory<FunctorT>(functor), n, r, a, b, c);
}

template<typename T>
struct MultiplyFunctor {
  __device__ T operator()(T x, T y) const {
    return x*y;
  }
};

template<>
struct MultiplyFunctor<half> {
  __device__ half operator()(half x, half y) const {
    return x*y;
  }
#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  __device__ void Apply2(half* z, const half* x, const half* y) const {
    const half2 x2 = *(reinterpret_cast<const half2*>(x));
    const half2 y2 = *(reinterpret_cast<const half2*>(y));
    *reinterpret_cast<half2*>(z) = __hmul2(x2, y2);
  }
#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
};

template<typename T>
struct AddFunctor {
  __device__ T operator()(T x, T y) const {
    return x+y;
  }
};

template<>
struct AddFunctor<half> {
  __device__ half operator()(half x, half y) const {
    return x*y;
  }
#if (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
  __device__ void Apply2(half* z, const half* x, const half* y) const {
    const half2 x2 = *(reinterpret_cast<const half2*>(x));
    const half2 y2 = *(reinterpret_cast<const half2*>(y));
    *reinterpret_cast<half2*>(z) = __hadd2(x2, y2);
  }
#endif  // (__CUDA_ARCH__ >= 750 && CUDA_VERSION >= 11000)
};

template<typename T>
__global__ void mul_naive(T *x, T *y, T* z, const int64_t N) {
    int g_tid = blockDim.x*blockIdx.x+threadIdx.x;
    for(int i=g_tid; i<N; i+=blockDim.x*gridDim.x) {
        z[i] = x[i] * y[i];
    }
}

template<typename T>
__global__ void add_naive(T *x, T *y, T* z, const int64_t N) {
    int g_tid = blockDim.x*blockIdx.x+threadIdx.x;
    for(int i=g_tid; i<N; i+=blockDim.x*gridDim.x) {
        z[i] = x[i] + y[i];
    }
}

/*
template<int64_t N>
__global__ void mul_naive(half *x, half *y, half* z){
    int g_tid = blockDim.x*blockIdx.x+threadIdx.x;
    for(int i=g_tid; i<N; i+=blockDim.x*gridDim.x) {
        z[i] = x[i] * y[i];
    }
}
*/

template<typename T>
__global__ void mul(T *x, T *y, T* z){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  z[idx] = x[idx] * y[idx];
}

template<>
__global__ void mul(half *x, half *y, half* z){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  z[idx] = x[idx] * y[idx];
}

template<typename Dtype>
class ElementWiseMul {
public:
    ElementWiseMul(int64_t num_elements, int iters, const size_t band_width, 
                   int fixed_grid_dim=500, int fixed_block_dim=256):
        __num_elements(num_elements), __iters(iters), 
        __band_width(band_width), __opt_grid_dim(1),
        __fixed_grid_dim(fixed_grid_dim), __fixed_block_dim(kBlockSize),
        __x_host(nullptr), __y_host(nullptr), __output_host(nullptr),
        __x_device(nullptr), __y_device(nullptr), __output_device(nullptr) {

        __malloc_cpy();

        cudaStreamCreate(&__stream);
        cudaEventCreate(&__start);
        cudaEventCreate(&__stop);
    }

    ~ElementWiseMul() {
        free(__x_host);
        free(__y_host);
        free(__output_host);
        cudaFree(__x_device);
        cudaFree(__y_device);
        cudaFree(__output_device);
    }


    float elementwise_mul_naive(const int grid_dim) {
        float vec_mul_time = 0.0f;

        cudaEventRecord(__start, __stream);
        for(int i=0; i<__iters; ++i) 
            mul_naive<Dtype><<<grid_dim, __fixed_block_dim, 0, __stream>>>(
                         __x_device, __y_device, 
                         __output_device, __num_elements);

        cudaEventRecord(__stop, __stream);
        cudaStreamSynchronize(__stream);
        cudaEventElapsedTime(&vec_mul_time, __start, __stop);

        vec_mul_time = vec_mul_time/__iters;
        float efficiency = __calc_efficiency(vec_mul_time,
                                            grid_dim, __fixed_block_dim, 
                                            true);

        //cudaMemcpyAsync(__output_host, __output_device, 
        //                __num_elements*sizeof(Dtype), cudaMemcpyDeviceToHost, __stream);

        //cudaStreamSynchronize(__stream);
        return efficiency;
    }

    float elementwise_mul_optimized() {
        float vec_mul_time = 0.0f;

        cudaEventRecord(__start);
        for(int i=0; i<__iters; ++i) 
            Binary(MultiplyFunctor<Dtype>(), &__opt_grid_dim, __num_elements, 
                   __output_device, __x_device, __y_device);
        cudaEventRecord(__stop);
        cudaEventSynchronize(__stop);
        cudaEventElapsedTime(&vec_mul_time, __start, __stop);

        vec_mul_time = vec_mul_time/__iters;
        float efficiency = __calc_efficiency(vec_mul_time,
                                            __opt_grid_dim, __fixed_block_dim, 
                                            true);

        //cudaMemcpy(__output_host, __output_device, 
        //           __num_elements*sizeof(Dtype), cudaMemcpyDeviceToHost);

        return efficiency;
    }

    float elementwise_add_naive(const int grid_dim) {
        float vec_add_time = 0.0f;

        cudaEventRecord(__start, __stream);
        for(int i=0; i<__iters; ++i) 
            add_naive<Dtype><<<grid_dim, __fixed_block_dim, 0, __stream>>>(
                         __x_device, __y_device, 
                         __output_device, __num_elements);

        cudaEventRecord(__stop, __stream);
        cudaStreamSynchronize(__stream);
        cudaEventElapsedTime(&vec_add_time, __start, __stop);

        vec_add_time = vec_add_time/__iters;
        float efficiency = __calc_efficiency(vec_add_time,
                                            grid_dim, __fixed_block_dim, 
                                            true);

        //cudaMemcpyAsync(__output_host, __output_device, 
        //                __num_elements*sizeof(Dtype), cudaMemcpyDeviceToHost, __stream);

        //cudaStreamSynchronize(__stream);
        return efficiency;
    }

    float elementwise_add_optimized() {
        float vec_add_time = 0.0f;

        cudaEventRecord(__start);
        for(int i=0; i<__iters; ++i) 
            Binary(AddFunctor<Dtype>(), &__opt_grid_dim, __num_elements, 
                   __output_device, __x_device, __y_device);
        cudaEventRecord(__stop);
        cudaEventSynchronize(__stop);
        cudaEventElapsedTime(&vec_add_time, __start, __stop);

        vec_add_time = vec_add_time/__iters;
        float efficiency = __calc_efficiency(vec_add_time,
                                            __opt_grid_dim, __fixed_block_dim, 
                                            true);

        //cudaMemcpy(__output_host, __output_device, 
        //           __num_elements*sizeof(Dtype), cudaMemcpyDeviceToHost);

        return efficiency;
    }

    inline int get_opt_grid_dim() {return __opt_grid_dim;}

private:
    void __malloc_cpy() {
        __x_host = reinterpret_cast<Dtype*>(malloc(__num_elements*sizeof(Dtype)));
        cudaMalloc((void **)&__x_device, __num_elements*sizeof(Dtype));
        for (int i = 0; i < __num_elements; i++) __x_host[i] = 2.0;
        cudaMemcpy(__x_device, __x_host, 
                   __num_elements*sizeof(Dtype), cudaMemcpyHostToDevice);

        __y_host = reinterpret_cast<Dtype*>(malloc(__num_elements*sizeof(Dtype)));
        cudaMalloc((void **)&__y_device, __num_elements*sizeof(Dtype));
        for (int i = 0; i < __num_elements; i++) __y_host[i] = 2.0;
        cudaMemcpy(__y_device, __y_host, __num_elements*sizeof(Dtype), cudaMemcpyHostToDevice);

        __output_host = reinterpret_cast<Dtype*>(malloc(__num_elements*sizeof(Dtype)));
        cudaMalloc((void **)&__output_device, __num_elements*sizeof(Dtype));
    }

    float __calc_efficiency(float calc_time,
                          int grid_dim, int block_dim, 
                          bool show_info=false) {

        const float inst_time = 0.f;
        size_t data_amount = (2+1)*sizeof(Dtype)*__num_elements;

        float estimate_time = (inst_time+(float)data_amount/__band_width/1024/1024/1024)*1000;
        float theoretical_throughput = (float)__num_elements/estimate_time;

        float throughput = (float)__num_elements/calc_time;
        float throughput_eff = throughput/theoretical_throughput;
        if(show_info) {
            printf("num_elements=%ld, grid_dim=%d, block_dim=%d, calc_time=%fms, estimate_time=%fms, throughput=%f, theoretical_throughput=%f, throughput_eff=%f \n", 
               __num_elements, grid_dim, block_dim,
               calc_time, estimate_time, 
               throughput, theoretical_throughput, throughput_eff );
        }

        return throughput_eff;
    }

private:
    const int64_t __num_elements;
    const size_t __band_width;
    int __opt_grid_dim;

    int __iters;
    int __fixed_grid_dim;
    int __fixed_block_dim;

    Dtype *__x_host;
    Dtype *__y_host;
    Dtype *__output_host;

    Dtype *__x_device;
    Dtype *__y_device;
    Dtype *__output_device;

    cudaStream_t __stream;
    cudaEvent_t __start, __stop;
};

float elementwise_mul_half_naive(const int64_t num_elements, 
                                 int iters, int grid_dim) {
    ElementWiseMul<half> mul(num_elements, iters, BAND_WIDTH);
    return mul.elementwise_mul_naive(grid_dim);
}

float elementwise_mul_float_naive(const int64_t num_elements,
                                  int iters, int grid_dim) {
    ElementWiseMul<float> mul(num_elements, iters, BAND_WIDTH);
    return mul.elementwise_mul_naive(grid_dim);
}

std::pair<int, float> elementwise_mul_half_optimized(const int64_t num_elements,
                                                     int iters) {
    ElementWiseMul<half> mul(num_elements, iters, BAND_WIDTH);
    float eff = mul.elementwise_mul_optimized();
    int opt_grid_dim = mul.get_opt_grid_dim();

    return std::make_pair(opt_grid_dim, eff);
}


std::pair<int, float>elementwise_mul_float_optimized(const int64_t num_elements,
                                                    int iters) {
    ElementWiseMul<float> mul(num_elements, iters, BAND_WIDTH);
    float eff = mul.elementwise_mul_optimized();
    int opt_grid_dim = mul.get_opt_grid_dim();

    return std::make_pair(opt_grid_dim, eff);
}

float elementwise_add_half_naive(const int64_t num_elements, 
                                 int iters, int grid_dim) {
    ElementWiseMul<half> add(num_elements, iters, BAND_WIDTH);
    return add.elementwise_add_naive(grid_dim);
}

float elementwise_add_float_naive(const int64_t num_elements,
                                  int iters, int grid_dim) {
    ElementWiseMul<float> add(num_elements, iters, BAND_WIDTH);
    return add.elementwise_add_naive(grid_dim);
}

std::pair<int, float> elementwise_add_half_optimized(const int64_t num_elements,
                                                     int iters) {
    ElementWiseMul<half> add(num_elements, iters, BAND_WIDTH);
    float eff = add.elementwise_add_optimized();
    int opt_grid_dim = add.get_opt_grid_dim();

    return std::make_pair(opt_grid_dim, eff);
}


std::pair<int, float>elementwise_add_float_optimized(const int64_t num_elements,
                                                    int iters) {
    ElementWiseMul<float> add(num_elements, iters, BAND_WIDTH);
    float eff = add.elementwise_add_optimized();
    int opt_grid_dim = add.get_opt_grid_dim();

    return std::make_pair(opt_grid_dim, eff);
}

