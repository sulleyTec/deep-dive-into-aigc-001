#include "tensor.hpp"
#include "normalization.hpp"
#include <iostream>
#include <time.h>


float* rand_gen(uint32_t len) {
    // srand((unsigned)time(NULL));

    float *float_arr = (float*)malloc(len*sizeof(float));
    for(uint32_t i=0; i<len; ++i) {
        /* 
            rand() % (b-a)) + a         ==> [a, b)
            rand() % (b-a+1)) + a       ==> [a, b]
            rand() % (b - a)) + a + 1   ==> (a, b]
            rand() / float(RAND_MAX)    ==> [0, 1]
         */

        float_arr[i] = rand()/float(RAND_MAX);
    }

    return float_arr;
}

void test_vec_add(BackEnd backend) {
    uint32_t bs = 8ul;
    uint32_t dim1 = 32ul;
    uint32_t dim2 = 128ul;
    uint32_t dim3 = 128ul;

    uint32_t size_a = bs*dim1*dim2*dim3;
    uint32_t size_b = dim2*dim3;

    std::vector<uint32_t> shape_a = {bs, dim1, dim2, dim3};
    std::vector<uint32_t> shape_b = {dim2, dim3};

    float *arr = rand_gen(size_a);
    float *brr = rand_gen(size_b);

    geefer::Tensor<float> ta(arr, shape_a, backend);
    geefer::Tensor<float> tb(brr, shape_b, backend);

    geefer::Tensor<float> td = tb+ta;
    std::cout << "element wise add:" << td.shape_string() << std::endl;
}

int main() {

    /*
    uint32_t bs = 1ul;
    uint32_t dim1 = 1ul;
    uint32_t dim2 = 2;
    uint32_t dim3 = 3;
    uint32_t dim4 = 2;

    //float *arr = rand_gen(bs*dim1*dim2*dim3);
    //float *brr = rand_gen(bs*dim1*dim3*dim4);

    //float arr[] = {0.,1.,2.,3.,4.,5.};
    //float brr[] = {0.,1.,2.,3.,4.,5.};

    float arr[] = {1.1, 0.54, 2.32, 0.05, -1.23, -0.645};
    float brr[] = {-2.31, 0.35, 1.342, -1.432, 0.34, 1.2345};

    std::vector<uint32_t> shape_a = {bs, dim1, dim2, dim3};
    std::vector<uint32_t> shape_b = {bs, dim1, dim3, dim4};

    geefer::Tensor<float> ta(arr, shape_a);
    geefer::Tensor<float> tb(brr, shape_b);

    geefer::Tensor<float> tc = ta.mm(tb);
    std::cout << tc.shape_string() << std::endl;

    uint32_t size = tc.size();
    const float* tc_data = tc.cpu_data();
    std::cout << "size=" << size << std::endl;

    float sum = 0.f;
    for(uint32_t i=0; i<size; ++i) {
        std::cout << tc_data[i] << std::endl;
        sum += tc_data[i];
    }

    std::cout << "sum=" << sum << std::endl;
    */

    /*
    //float *arr = rand_gen(bs*dim1*dim2*dim3);
    float arr[] = {1.,2.,3.,4.,5.,6.};
    // (2,8,32,16)
    //std::vector<uint32_t> shape_a = {bs, dim1, dim3, dim4};
    std::vector<uint32_t> shape_a = {1, 1, 2, 3};
    geefer::Tensor<float> ta(arr, shape_a);
    std::cout << "before:" << ta.shape_string() << std::endl;

    // (1,2,3,1)
    std::vector<uint32_t> perm= {1,2,3,0};
    ta.Permute(perm);
    std::cout << "after:" << ta.shape_string() << std::endl;
    const float *data = ta.cpu_data();

    for(uint32_t i=0; i<ta.size(); ++i) {
        std::cout << data[i] << ", ";
    }
    std::cout << std::endl;
    */

    /*
    uint32_t bs = 2ul;
    uint32_t dim1 = 8ul;
    uint32_t dim2 = 32ul;
    uint32_t dim3 = 32ul;
    uint32_t dim4 = 16ul;
    */

    /*
    uint32_t bs = 2ul;
    uint32_t dim1 = 1ul;
    uint32_t dim2 = 3ul;
    uint32_t dim3 = 4ul;
    uint32_t dim4 = 4ul;

    std::vector<uint32_t> shape_a = {bs, dim1, dim2, dim4};

    //float *arr = rand_gen(bs*dim1*dim2*dim4);
    float arr[24];

    for(int i=0; i<24; ++i){
        arr[i] = static_cast<float>(i);
    }

    geefer::Tensor<float> ta(arr, shape_a);

    std::vector<int32_t> reduce_axes = {0,-2};
    //geefer::Tensor<float> mean = ta.Mean(reduce_axes);
    geefer::Tensor<float> var = ta.Var(reduce_axes);
    std::cout << var.shape_string() << std::endl;
    */

    /*
    uint32_t bs = 6ul;
    uint32_t dim1 = 3ul;
    uint32_t dim2 = 4ul;
    uint32_t dim3 = 5ul;

    uint32_t size_a = bs*dim1*dim2*dim3;
    std::vector<uint32_t> shape_a = {bs, dim1, dim2};
    std::vector<uint32_t> shape_b = {bs, dim1, dim2, dim3};
    float *arr = rand_gen(size_a);
    //geefer::Tensor<float> ta(arr, shape_a, GPU);
    geefer::Tensor<float> ta(arr, shape_a);
    geefer::Tensor<float> tb(arr, shape_b);

    std::vector<uint32_t> new_shape = {bs, dim1, dim2, 1};
    ta.Reshape(new_shape);
    std::cout << "ta.shape=: " << ta.shape_string() << std::endl;
    std::cout << "tb.shape=: " << tb.shape_string() << std::endl;

    geefer::Tensor<float> tc = tb - ta;
    std::cout << "minus: " << tc.shape_string() << std::endl;

    //uint32_t size_b = dim1*dim3;
    //std::vector<uint32_t> shape_b = {dim1, 1, dim3};
    //float *brr = rand_gen(size_b);
    //geefer::Tensor<float> tb(brr, shape_b, GPU);

    //geefer::Tensor<float> td = tb+ta;
    //std::cout << "element wise add:" << td.shape_string() << std::endl;
    */

    /*

    uint32_t size_b = dim1*dim3;
    std::vector<uint32_t> shape_b = {dim1, 1, dim3};
    float *brr = rand_gen(size_b);
    geefer::Tensor<float> tb(brr, shape_b);

    //geefer::Tensor<float> tc = ta*tb;
    geefer::Tensor<float> tc = tb*ta;
    std::cout << "element wise multiply:" << tc.shape_string() << std::endl;

    geefer::Tensor<float> td = tb+ta;
    std::cout << "element wise add:" << td.shape_string() << std::endl;

    geefer::Tensor<float> te = tb/ta;
    std::cout << "element wise divide:" << te.shape_string() << std::endl;

    geefer::Tensor<float> tf = tb-ta;
    std::cout << "element wise minus:" << tf.shape_string() << std::endl;

    geefer::Tensor<float> tg = ta.Sqrt();
    std::cout << "element wise sqrt:" << tf.shape_string() << std::endl;
    */

    /*
    uint32_t bs = 20ul;
    uint32_t dim1 = 5ul;
    uint32_t dim2 = 20ul;
    uint32_t dim3 = 10ul;

    uint32_t size_a = bs*dim1*dim2*dim3;
    std::vector<uint32_t> shape_a = {bs, dim1, dim2, dim3};
    float *arr = rand_gen(size_a);
    geefer::Tensor<float> ta(arr, shape_a);

    geefer::Tensor<float> result = ta+1e-3f;
    std::cout << result.shape_string() << std::endl;

    geefer::LayerNorm<float> layernorm({dim3}, ta.shape());
    geefer::Tensor<float> result = layernorm.forward(ta);
    std::cout << result.shape_string() << std::endl;
    */

    test_vec_add(GPU);

    return 0;
}

