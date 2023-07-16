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
    explicit Normal(uint32_t idx, const std::vector<uint32_t>& shape):
        _norm_idx(idx), 
        _init_gamma_val(static_cast<DType>(1.0)), 
        _init_beta_val(static_cast<DType>(0.0)),
        _shape(shape)  {}

    virtual ~Normal() {};

    virtual inline Tensor<DType> forward(const Tensor<DType> &input)=0;

protected:
    virtual inline Tensor<DType> _mean(const Tensor<DType>& input) const =0;
    virtual inline Tensor<DType> _var(const Tensor<DType>& input) const =0;
    virtual inline Tensor<DType> _init_param(DType init_v)=0;

    inline void _init_gamma() {
        _gamma = _init_param(_init_gamma_val);
    }

    inline void _init_beta() {
        _beta = _init_param(_init_beta_val);
    }

protected:
    uint32_t _norm_idx;
    DType _init_gamma_val;
    DType _init_beta_val;

    Tensor<DType> _gamma;
    Tensor<DType> _beta;

    std::vector<uint32_t> _shape;

}; // class Normal


/* 4D input */
/*
template<typename DType>
class BatchNorm: public Normal
{
public:
    BatchNorm() = default;
    explicit BatchNorm(const Tensor<DType> &input): 
        Normal(1, input) {}

    virtual inline void forward() {

    }

protected:
    virtual inline void _mean() {

    }

    virtual inline void _var() {

    }

};
*/

/* 4D input */
template<typename DType>
class LayerNorm: public Normal<DType>
{
public:
    LayerNorm() = default;
    explicit LayerNorm(const std::vector<uint32_t> &shape):
        Normal<DType>(3, shape) {
            _init_gamma();
            _init_beta();
        }

    Tensor<DType> forward(const Tensor<DType> &input) override {
        const Tensor<DType> mean_vec = _mean(input);
        const Tensor<DType> var_vec = _var(input);

        //const Tensor<DType> tmp = (input - mean_vec)/(var_vec.Sqrt());
        //return tmp * _gamma + _beta;

        return (input - mean_vec)/(var_vec.Sqrt())*_gamma+_beta;
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
        //std::vector<int32_t> mean_axes = {_norm_idx};
        return input.Mean({_norm_idx});
    }

    inline Tensor<DType>_var(const Tensor<DType>& input) const {
        //std::vector<int32_t> var_axes = {_norm_idx};
        return input.Var({_norm_idx});
    }

protected:
    using Normal<DType>::_norm_idx;
    using Normal<DType>::_init_gamma_val;
    using Normal<DType>::_init_beta_val;
    using Normal<DType>::_gamma;
    using Normal<DType>::_beta;
    using Normal<DType>::_shape;

};

}; // namespace geefer

#endif // __SYNCMEM__

