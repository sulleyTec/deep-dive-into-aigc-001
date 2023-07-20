#ifndef __NORM_HPP__
#define __NORM_HPP__

#include "common.hpp"
#include "tensor.hpp"

namespace geefer {

template<typename DType>
class Normal
{
public:
    Normal() = default;
    explicit Normal(const std::vector<uint32_t> &norm_shape, 
                    const std::vector<uint32_t> &shape):
        _norm_axes(norm_shape.size()),
        _init_gamma_val(static_cast<DType>(1.0)), 
        _init_beta_val(static_cast<DType>(0.0)),
        _shape(shape),
        _norm_shape(norm_shape) {

    }

    virtual ~Normal() {};

    virtual inline Tensor<DType> forward(const Tensor<DType> &input)=0;

protected:
    virtual inline Tensor<DType> _mean(const Tensor<DType>& input) const =0;
    virtual inline Tensor<DType> _var(const Tensor<DType>& input, 
                                      uint32_t correction=1) const =0;
    virtual inline Tensor<DType> _init_param(DType init_v)=0;

    inline void _init_gamma() {
        _gamma = _init_param(_init_gamma_val);
    }

    inline void _init_beta() {
        _beta = _init_param(_init_beta_val);
    }

protected:
    std::vector<int32_t> _norm_axes;
    DType _init_gamma_val;
    DType _init_beta_val;

    Tensor<DType> _gamma;
    Tensor<DType> _beta;

    std::vector<uint32_t> _shape;
    std::vector<uint32_t> _norm_shape;

}; // class Normal


/* 4D input */
template<typename DType>
class LayerNorm: public Normal<DType>
{
public:
    LayerNorm() = default;
    explicit LayerNorm(const std::vector<uint32_t> &norm_shape, 
                       const std::vector<uint32_t> &shape):
        Normal<DType>(norm_shape, shape) {
            int32_t i=norm_shape.size()-1; 
            int32_t j=shape.size()-1;

            /* compare norm axes in reverse order */
            for(; i>=0; --i) {
                if(norm_shape[i] != shape[j]) {
                    Warning("normalization shape is not correct");
                    exit(1);
                }
                _norm_axes[i] = j--;
            } // for loop

            _init_gamma();
            _init_beta();
        }

    Tensor<DType> forward(const Tensor<DType> &input) override {
        Tensor<DType> mean_vec = _mean(input);
        Tensor<DType> var_vec = _var(input, 0);

        std::vector<uint32_t> new_shape(input.axes_num()); 

        for(uint32_t i=0; i<input.axes_num()-_norm_axes.size(); ++i) {
            new_shape[i] = _shape[i];
        }

        for(uint32_t i=0; i<_norm_axes.size(); ++i) {
            new_shape[_norm_axes[i]] = 1;
        }

        mean_vec.Reshape(new_shape);
        var_vec.Reshape(new_shape);

        return (input - mean_vec)/((var_vec+static_cast<DType>(1e-3)).Sqrt())*_gamma+_beta;
        //return (input - mean_vec)/((var_vec+static_cast<DType>(1e-3)).Sqrt());
    }

protected:
    using Normal<DType>::_init_gamma;
    using Normal<DType>::_init_beta;

    inline Tensor<DType> _init_param(DType init_v) override {
        std::vector<uint32_t> param_shape(_shape.size()-1);

        uint32_t dim = 1;
        for(uint32_t i=0; i<param_shape.size(); ++i) {
            dim *= _shape[i+1];
            param_shape[i] = _shape[i+1];
        }

        std::vector<DType> param({dim}, init_v);
        return Tensor<DType>(param.data(), param_shape);
    }

    inline Tensor<DType> _mean(const Tensor<DType>& input) const {
        return input.Mean(_norm_axes);
    }

    inline Tensor<DType>_var(const Tensor<DType>& input,
                             uint32_t correction=1) const {
        return input.Var(_norm_axes, correction);
    }

protected:
    using Normal<DType>::_norm_axes;
    using Normal<DType>::_init_gamma_val;
    using Normal<DType>::_init_beta_val;
    using Normal<DType>::_gamma;
    using Normal<DType>::_beta;
    using Normal<DType>::_shape;
    using Normal<DType>::_norm_shape;
};

}; // namespace geefer

#endif // __SYNCMEM__

