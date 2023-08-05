#ifndef __ELEMENTWISE__
#define __ELEMENTWISE__

float elementwise_mul_half_naive(const int64_t num_elements, 
                                 int iters, int grid_dim=500);
float elementwise_mul_float_naive(const int64_t num_elements, 
                                  int iters, int grid_dim=500);
std::pair<int, float>elementwise_mul_half_optimized(const int64_t num_elements, 
                                     int iters);
std::pair<int, float>elementwise_mul_float_optimized(const int64_t num_elements, 
                                      int iters);

float elementwise_add_half_naive(const int64_t num_elements, 
                                 int iters, int grid_dim);

float elementwise_add_float_naive(const int64_t num_elements,
                                  int iters, int grid_dim);

std::pair<int, float> elementwise_add_half_optimized(const int64_t num_elements,
                                                     int iters);

std::pair<int, float>elementwise_add_float_optimized(const int64_t num_elements,
                                                    int iters);

#endif
