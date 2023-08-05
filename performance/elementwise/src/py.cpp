#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cuda_fp16.h>
#include "elementwise.hpp"

namespace py = pybind11;

float element_wise_mul_naive_half(const int64_t num_elements,
                                  int grid_dim, int iters=10) {
    return elementwise_mul_half_naive(num_elements, iters, grid_dim);
}

float element_wise_mul_naive_float(const int64_t num_elements,
                                   int grid_dim, int iters=10) {
    return elementwise_mul_float_naive(num_elements, iters, grid_dim);
}

std::pair<int, float>element_wise_mul_optimized_half(const int64_t num_elements,
                                      int iters=10) {
    return elementwise_mul_half_optimized(num_elements, iters);
}

std::pair<int, float>element_wise_mul_optimized_float(const int64_t num_elements,
                                       int iters=10) {
    return elementwise_mul_float_optimized(num_elements, iters);
}

PYBIND11_MODULE(libvecmul, m) {
    m.def("element_wise_mul_naive_half", &element_wise_mul_naive_half);
    m.def("element_wise_mul_optimized_half", &element_wise_mul_optimized_half);
    m.def("element_wise_mul_naive_float", &element_wise_mul_naive_float);
    m.def("element_wise_mul_optimized_float", &element_wise_mul_optimized_float);
}

