#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

#include <string.h>
#include <iostream>
#include <string>
#include "tensor.hpp"
#include "normalization.hpp"

namespace py = pybind11;

py::array_t<float> gf_layernorm(const py::array_t<float>& input,
                                const std::vector<uint32_t> &norm_shape) {

    py::buffer_info input_buf = input.request();
    std::vector<uint32_t> shape(input_buf.ndim);

    for(uint32_t i=0; i<input_buf.ndim; ++i) {
        shape[i] = input_buf.shape[i];
    }

    float* arr = static_cast<float*>(input_buf.ptr);
    geefer::Tensor<float> ta(arr, shape);
    geefer::LayerNorm<float> layernorm(norm_shape, ta.shape());
    geefer::Tensor<float> m_res = layernorm.forward(ta);

    py::array_t<float> result_array(m_res.shape());
    py::buffer_info result_buf_info = result_array.request();

    float* result_data = static_cast<float*>(result_buf_info.ptr);
    const float* m_data = m_res.cpu_data();

    memcpy(result_data, m_data, m_res.size()*sizeof(float));

    return result_array;
}

py::array_t<float> gf_element_wise_sqrt(const py::array_t<float>& input) {

    py::buffer_info input_buf = input.request();

    std::vector<uint32_t> shape(input_buf.ndim);

    for(uint32_t i=0; i<input_buf.ndim; ++i) {
        shape[i] = input_buf.shape[i];
    }

    float* arr = static_cast<float*>(input_buf.ptr);
    geefer::Tensor<float> ta(arr, shape);
    geefer::Tensor<float> m_res = ta.Sqrt();

    py::array_t<float> result_array(m_res.shape());
    py::buffer_info result_buf_info = result_array.request();

    float* result_data = static_cast<float*>(result_buf_info.ptr);
    const float* m_data = m_res.cpu_data();

    memcpy(result_data, m_data, m_res.size()*sizeof(float));

    return result_array;
}

py::array_t<float> gf_element_wise_minus(const py::array_t<float>& input1,
                                         const py::array_t<float>& input2) {

    py::buffer_info input_buf1 = input1.request();
    py::buffer_info input_buf2 = input2.request();

    std::vector<uint32_t> shape_a(input_buf1.ndim);
    std::vector<uint32_t> shape_b(input_buf2.ndim);

    for(uint32_t i=0; i<input_buf1.ndim; ++i) {
        shape_a[i] = input_buf1.shape[i];
        shape_b[i] = input_buf2.shape[i];
    }

    float* arr = static_cast<float*>(input_buf1.ptr);
    geefer::Tensor<float> ta(arr, shape_a);

    float* brr = static_cast<float*>(input_buf2.ptr);
    geefer::Tensor<float> tb(brr, shape_b);

    geefer::Tensor<float> m_res = ta-tb;

    py::array_t<float> result_array(m_res.shape());
    py::buffer_info result_buf_info = result_array.request();

    float* result_data = static_cast<float*>(result_buf_info.ptr);
    const float* m_data = m_res.cpu_data();

    memcpy(result_data, m_data, m_res.size()*sizeof(float));

    return result_array;
}

py::array_t<float> gf_element_wise_div(const py::array_t<float>& input1,
                                       const py::array_t<float>& input2) {

    py::buffer_info input_buf1 = input1.request();
    py::buffer_info input_buf2 = input2.request();

    std::vector<uint32_t> shape_a(input_buf1.ndim);
    std::vector<uint32_t> shape_b(input_buf2.ndim);

    for(uint32_t i=0; i<input_buf1.ndim; ++i) {
        shape_a[i] = input_buf1.shape[i];
        shape_b[i] = input_buf2.shape[i];
    }

    float* arr = static_cast<float*>(input_buf1.ptr);
    geefer::Tensor<float> ta(arr, shape_a);

    float* brr = static_cast<float*>(input_buf2.ptr);
    geefer::Tensor<float> tb(brr, shape_b);

    geefer::Tensor<float> m_res = ta/tb;

    py::array_t<float> result_array(m_res.shape());
    py::buffer_info result_buf_info = result_array.request();

    float* result_data = static_cast<float*>(result_buf_info.ptr);
    const float* m_data = m_res.cpu_data();

    memcpy(result_data, m_data, m_res.size()*sizeof(float));

    return result_array;
}

py::array_t<float> gf_element_wise_add_scalar(const py::array_t<float>& input1,
                                              const float input2) {

    py::buffer_info input_buf1 = input1.request();

    std::vector<uint32_t> shape_a(input_buf1.ndim);

    for(uint32_t i=0; i<input_buf1.ndim; ++i) {
        shape_a[i] = input_buf1.shape[i];
    }

    float* arr = static_cast<float*>(input_buf1.ptr);
    geefer::Tensor<float> ta(arr, shape_a);

    geefer::Tensor<float> m_res = ta+input2;

    py::array_t<float> result_array(m_res.shape());
    py::buffer_info result_buf_info = result_array.request();

    float* result_data = static_cast<float*>(result_buf_info.ptr);
    const float* m_data = m_res.cpu_data();

    memcpy(result_data, m_data, m_res.size()*sizeof(float));

    return result_array;
}

py::array_t<float> gf_element_wise_add(const py::array_t<float>& input1,
                                       const py::array_t<float>& input2,
                                       BackEnd backend=CPU) {

    py::buffer_info input_buf1 = input1.request();
    py::buffer_info input_buf2 = input2.request();

    std::vector<uint32_t> shape_a(input_buf1.ndim);
    std::vector<uint32_t> shape_b(input_buf2.ndim);

    for(uint32_t i=0; i<input_buf1.ndim; ++i) {
        shape_a[i] = input_buf1.shape[i];
        shape_b[i] = input_buf2.shape[i];
    }

    float* arr = static_cast<float*>(input_buf1.ptr);
    geefer::Tensor<float> ta(arr, shape_a, backend);

    float* brr = static_cast<float*>(input_buf2.ptr);
    geefer::Tensor<float> tb(brr, shape_b, backend);

    geefer::Tensor<float> m_res = ta+tb;

    py::array_t<float> result_array(m_res.shape());
    py::buffer_info result_buf_info = result_array.request();

    float* result_data = static_cast<float*>(result_buf_info.ptr);
    const float* m_data = m_res.cpu_data();

    memcpy(result_data, m_data, m_res.size()*sizeof(float));

    return result_array;
}

py::array_t<float> gf_element_wise_mul(const py::array_t<float>& input1,
                                       const py::array_t<float>& input2) {

    py::buffer_info input_buf1 = input1.request();
    py::buffer_info input_buf2 = input2.request();

    std::vector<uint32_t> shape_a(input_buf1.ndim);
    std::vector<uint32_t> shape_b(input_buf2.ndim);

    for(uint32_t i=0; i<input_buf1.ndim; ++i) {
        shape_a[i] = input_buf1.shape[i];
        shape_b[i] = input_buf2.shape[i];
    }

    float* arr = static_cast<float*>(input_buf1.ptr);
    geefer::Tensor<float> ta(arr, shape_a);

    float* brr = static_cast<float*>(input_buf2.ptr);
    geefer::Tensor<float> tb(brr, shape_b);

    geefer::Tensor<float> m_res = ta*tb;

    py::array_t<float> result_array(m_res.shape());
    py::buffer_info result_buf_info = result_array.request();

    float* result_data = static_cast<float*>(result_buf_info.ptr);
    const float* m_data = m_res.cpu_data();

    memcpy(result_data, m_data, m_res.size()*sizeof(float));

    return result_array;
}

py::array_t<float> gf_broadcast(const std::vector<uint32_t>& axes,
                                const py::array_t<float>& input) {

    py::buffer_info input_buf = input.request();
    std::vector<uint32_t> shape_a(input_buf.ndim);

    for(uint32_t i=0; i<input_buf.ndim; ++i)
        shape_a[i] = input_buf.shape[i];

    float* arr = static_cast<float*>(input_buf.ptr);
    geefer::Tensor<float> ta(arr, shape_a);

    geefer::Tensor<float> m_res = ta.BroadCast(axes);

    py::array_t<float> result_array(m_res.shape());
    py::buffer_info result_buf_info = result_array.request();

    float* result_data = static_cast<float*>(result_buf_info.ptr);
    const float* m_data = m_res.cpu_data();

    memcpy(result_data, m_data, m_res.size()*sizeof(float));

    return result_array;
}

py::array_t<float> gf_var(const std::vector<int32_t>& reduced_axes,
                           const py::array_t<float>& input,
                           uint32_t correction=1) {

    py::buffer_info input_buf = input.request();
    std::vector<uint32_t> shape_a(input_buf.ndim);

    for(uint32_t i=0; i<input_buf.ndim; ++i)
        shape_a[i] = input_buf.shape[i];

    float* arr = static_cast<float*>(input_buf.ptr);
    geefer::Tensor<float> ta(arr, shape_a);

    geefer::Tensor<float> m_res = ta.Var(reduced_axes, correction);

    py::array_t<float> result_array(m_res.shape());
    py::buffer_info result_buf_info = result_array.request();

    float* result_data = static_cast<float*>(result_buf_info.ptr);
    const float* m_data = m_res.cpu_data();

    memcpy(result_data, m_data, m_res.size()*sizeof(float));

    return result_array;
}

void gf_reshape(py::array_t<float>& input, 
                const std::vector<uint32_t>& new_shape) {

    py::buffer_info input_buf = input.request();
    std::vector<uint32_t> shape_a(input_buf.ndim);

    for(uint32_t i=0; i<input_buf.ndim; ++i)
        shape_a[i] = input_buf.shape[i];

    float* arr = static_cast<float*>(input_buf.ptr);
    geefer::Tensor<float> ta(arr, shape_a);

    ta.Reshape(new_shape);
    std::cout << "reshape: " << ta.shape_string() << std::endl;
    std::cout << "axes_num: " << ta.axes_num() << std::endl;

    input.resize(ta.shape());
}

py::array_t<float> gf_mean(const std::vector<int32_t>& reduced_axes,
                           const py::array_t<float>& input) {

    py::buffer_info input_buf = input.request();
    std::vector<uint32_t> shape_a(input_buf.ndim);

    for(uint32_t i=0; i<input_buf.ndim; ++i)
        shape_a[i] = input_buf.shape[i];

    float* arr = static_cast<float*>(input_buf.ptr);
    geefer::Tensor<float> ta(arr, shape_a);

    geefer::Tensor<float> m_res = ta.Mean(reduced_axes);

    py::array_t<float> result_array(m_res.shape());
    py::buffer_info result_buf_info = result_array.request();

    float* result_data = static_cast<float*>(result_buf_info.ptr);
    const float* m_data = m_res.cpu_data();

    memcpy(result_data, m_data, m_res.size()*sizeof(float));

    return result_array;
}

py::array_t<float> gf_permute(const std::vector<uint32_t>& perm,
                               const py::array_t<float>& input) {

    py::buffer_info input_buf = input.request();
    std::vector<uint32_t> shape_a(input_buf.ndim);

    for(uint32_t i=0; i<input_buf.ndim; ++i)
        shape_a[i] = input_buf.shape[i];

    /* geefer mm */
    float* arr = static_cast<float*>(input_buf.ptr);
    geefer::Tensor<float> ta(arr, shape_a);

    ta.Permute(perm);

    py::array_t<float> result_array(ta.shape());
    py::buffer_info result_buf_info = result_array.request();

    float* result_data = static_cast<float*>(result_buf_info.ptr);
    const float* ta_data = ta.cpu_data();

    memcpy(result_data, ta_data, ta.size()*sizeof(float));

    /*
    std::cout << "result_data: " << std::endl;
    for(uint32_t i=0; i< ta.size(); ++i) {
        std::cout << result_data[i] << ", ";
    }
    std::cout << std::endl;
    */

    return result_array;
}


// Function that takes two NumPy arrays and returns their matrix multiplication
/*
py::array_t<float> matrix_multiply(py::array_t<float> array1, py::array_t<float> array2) {
    py::buffer_info buf_info1 = array1.request();
    py::buffer_info buf_info2 = array2.request();

    if (buf_info1.ndim != buf_info2.ndim) {
        throw std::runtime_error("Input arrays must have the same number of dimensions!");
    }

    if (buf_info1.ndim < 2) {
        throw std::runtime_error("Input arrays must have at least 2 dimensions!");
    }

    std::vector<uint32_t> result_shape;
    for (uint32_t i = 0; i < buf_info1.ndim - 2; ++i) {
        if (buf_info1.shape[i] != buf_info2.shape[i]) {
            throw std::runtime_error("Incompatible array dimensions for matrix multiplication!");
        }
        result_shape.push_back(buf_info1.shape[i]);
    }
    result_shape.push_back(buf_info1.shape[buf_info1.ndim - 2]);
    result_shape.push_back(buf_info2.shape[buf_info2.ndim - 1]);

    py::array_t<float> result_array(result_shape);
    py::buffer_info result_buf_info = result_array.request();
    float* result_data = static_cast<float*>(result_buf_info.ptr);

    float* data1 = static_cast<float*>(buf_info1.ptr);
    float* data2 = static_cast<float*>(buf_info2.ptr);

    uint32_t num_dims = buf_info1.ndim;
    uint32_t num_elements = 1;
    for (uint32_t i = 0; i < num_dims - 2; ++i) {
        num_elements *= buf_info1.shape[i];
    }
    uint32_t rows = buf_info1.shape[num_dims - 2];
    uint32_t cols = buf_info2.shape[num_dims - 1];
    uint32_t inner_dim = buf_info1.shape[num_dims - 1];

    for (uint32_t i = 0; i < num_elements; ++i) {
        for (uint32_t j = 0; j < rows; ++j) {
            for (uint32_t k = 0; k < cols; ++k) {
                float sum = 0.0;
                for (uint32_t m = 0; m < inner_dim; ++m) {
                    uint32_t offset1 = (i*rows+j)*inner_dim+m;
                    uint32_t offset2 = (i*inner_dim+m)*cols+k;
                    sum += data1[offset1] * data2[offset2];
                }

                uint32_t offset = (i*rows+j)*cols+k;
                result_data[offset] = sum;
            }
        }
    }

    return result_array;
}
*/

py::array_t<float> gf_matrix_multiply(const py::array_t<float> array1, 
                                      const py::array_t<float> array2) {

    py::buffer_info buf_info1 = array1.request();
    py::buffer_info buf_info2 = array2.request();

    float* arr = static_cast<float*>(buf_info1.ptr);
    float* brr = static_cast<float*>(buf_info2.ptr);

    uint32_t ndim = buf_info1.ndim;

    std::vector<uint32_t> shape_a(ndim);
    std::vector<uint32_t> shape_b(ndim);

    for(uint32_t i=0; i<buf_info1.ndim; ++i) {
        shape_a[i] = buf_info1.shape[i];
        shape_b[i] = buf_info2.shape[i];
    }


    /* geefer mm */
    geefer::Tensor<float> ta(arr, shape_a);
    geefer::Tensor<float> tb(brr, shape_b);

    geefer::Tensor<float> tc = ta.mm(tb);
    const float* tc_data = tc.cpu_data();
    uint32_t size = tc.size();

    /*
    std::cout << "tc.shape=" << tc.shape_string() << std::endl;
    std::cout << "tc.size=" << size << std::endl;
    std::cout << "tc_data: " << std::endl;
    for(uint32_t i = 0; i<size; ++i) {
        std::cout << tc_data[i] << ", ";
    }
    std::cout << std::endl;
    */

    std::vector<uint32_t> result_shape = tc.shape();
    py::array_t<float> result_array(result_shape);
    py::buffer_info result_buf_info = result_array.request();
    float* result_data = static_cast<float*>(result_buf_info.ptr);

    memcpy(result_data, tc_data, size*sizeof(float));
    return result_array;
}

/*
template <typename T>
void bind_permutation(py::module& m) {
    m.def("permutation", [](const std::vector<size_t>& perm, const py::array_t<T>& input) {
        std::vector<py::ssize_t> shape;
        for (ssize_t i = 0; i < input.ndim(); ++i) {
            shape.push_back(input.shape(i));
        }

        py::array_t<T> output = py::array_t<T>(shape);
        permute_recursive(perm, 0, input, output);
        return output;
    }, "permute high dimensional matrix");
}
PYBIND11_MODULE(libgfinfer, m) {
    //bind_permutation<double>(m);
}*/

PYBIND11_MODULE(libgfinfer, m) {
    py::enum_<BackEnd>(m, "BackEnd")
        .value("CPU", BackEnd::CPU)
        .value("GPU", BackEnd::GPU)
        .export_values();
    m.def("matrix_multiply", &gf_matrix_multiply);
    m.def("permute", &gf_permute);
    m.def("reshape", &gf_reshape);
    m.def("mean", &gf_mean);
    m.def("var", &gf_var);
    m.def("broadcast", &gf_broadcast);
    m.def("element_wise_add", &gf_element_wise_add);
    m.def("element_wise_add_scalar", &gf_element_wise_add_scalar);
    m.def("element_wise_minus", &gf_element_wise_minus);
    m.def("element_wise_mul", &gf_element_wise_mul);
    m.def("element_wise_div", &gf_element_wise_div);
    m.def("element_wise_sqrt", &gf_element_wise_sqrt);
    m.def("layernorm", &gf_layernorm);
}

