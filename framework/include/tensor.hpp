#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include "syncmem.hpp"
#include "op.hpp"

// geefer for geek infer
namespace geefer
{

template<typename DType>
class Tensor {
public:
    Tensor(): _data(), _size(0ul), _capacity(0ul), _max_dim(4), _backend(CPU) 
    {}

    explicit Tensor(uint32_t num, uint32_t channels, 
                    uint32_t width, uint32_t height,
                    BackEnd backend=CPU):
        _capacity(0ul), _max_dim(4), _backend(backend)  {
        Reshape(num, channels, width, height);
    }

    explicit Tensor(const std::vector<uint32_t> &shape,
                    BackEnd backend=CPU):
        _capacity(0ul), _max_dim(4), _backend(backend) {
        Reshape(shape);
    }

    /* user MUST make sure that shape is correspond to len of data */
    explicit Tensor(DType* init_data, 
                    std::vector<uint32_t> &shape, 
                    BackEnd backend=CPU):
        _capacity(0ul), _max_dim(4), _backend(backend) {
        Reshape(shape);
        if(_backend==CPU)
            set_cpu_data(init_data);

        if(_backend==GPU)
            set_gpu_data(init_data);
    }

    /* copy construct */
    Tensor(const Tensor<DType> &other):
        _size(other._size),
        _capacity(other._capacity),
        _max_dim(4), _backend(other._backend) {

        _shape = other.shape();
        _size = other.size();
        _capacity = _size;
        _shape_data.reset(new SyncMem(_shape.size() * sizeof(uint32_t)));
        _data.reset(new SyncMem(_capacity*sizeof(DType)));

        CopyFrom(other);
    }

    //~Tensor() {}

    void Reshape(uint32_t num, uint32_t channels, 
                 uint32_t width, uint32_t height) {
        Reshape({num, channels, width, height});
    }

    void Reshape(const std::vector<uint32_t> &shape) {
        if(!CHECK_LE(shape.size(), kMaxTensorAxes)) {
            std::ostringstream stream;
            stream << std::to_string(shape.size()) << "out of range for ";
            Warning(stream.str());
        }

        _size = 1ul;
        _shape.resize(shape.size());

        /* */
        if(!_shape_data.get() || _shape_data->size() < shape.size()*sizeof(uint32_t)){
            _shape_data.reset(new SyncMem(shape.size() * sizeof(uint32_t)));
        }

        /*  */
        uint32_t* raw_shape = static_cast<uint32_t*>(_shape_data->mutable_cpu_data());

        for(uint32_t i = 0; i< shape.size(); ++i) {
            if(_size > 0) {
                if(CHECK_GT(shape[i], UINT32_MAX/_size)) {
                    std::ostringstream stream;
                    stream << std::to_string(shape[i]) << "out of range";
                    Warning(stream.str());
                    return;
                }
            }

            _size *= shape[i];
            _shape[i] = shape[i];
            raw_shape[i] = shape[i];
        }

        /* 
           only first time when create a new tensor,
           if _data alread had data, code will not enter in here
         */
        if(_size > _capacity) {
            _capacity = _size;
            _data.reset(new SyncMem(_size*sizeof(DType)));
        }
    }

    void ReshapeLike(const Tensor<DType>& other) {
        Reshape(other.shape());
    }

    Tensor<DType>& Permute(const std::vector<uint32_t> &perm) {
        if(perm.size()==4) {
            return __permute4(perm);
        }
        else if(perm.size()==3) {
            return __permute3(perm);
        }
        else if(perm.size()==2) {
            return __permute2(perm);
        }

        Warning("perm dimention is incorrect");
        exit(1);
    }

    /*
       axes is the target shape
       suppose axes=(6,3,4,5); _shape=(3,1,5)
    */
    Tensor<DType> BroadCast(const std::vector<uint32_t> axes) const {
        /* 
           validate shape: 
           compare shape in reverse order
         */
        if(axes.size() < _shape.size()) {
            Warning("broadcast shape does not align");
            exit(1);
        }

        std::vector<uint32_t> result_shape(axes.size(), 1);
        uint32_t j = axes.size()-1;

        for(int32_t i=_shape.size()-1; i>=0; --i) {

            if(_shape[i]==1) {j--; continue;}
            if(_shape[i]==axes[j]) {
                result_shape[j] = _shape[i];
                j--;
                continue;
            }
            else {
                Warning("broadcast shape does not align");
                exit(1);
            }
        }

        std::vector<uint32_t> perm;
        std::vector<uint32_t> tmp_shape(axes.size());
        std::vector<uint32_t> broadcast_axes;
        std::vector<uint32_t> reserved_axes;
        uint32_t broadcast_num = 1;

        for(uint32_t i=0; i<axes.size(); ++i) {
            if(result_shape[i]==1) {
                broadcast_axes.push_back(i);
                broadcast_num *= axes[i];
            }
            else 
                reserved_axes.push_back(i);
        }

        // suppose axes=(2,1,3,4); _shape=(1,4)
        // perm = (0,2,1,3)
        perm.insert(perm.end(),broadcast_axes.begin(),broadcast_axes.end());
        perm.insert(perm.end(),reserved_axes.begin(),reserved_axes.end());

        for(uint32_t i=0; i<tmp_shape.size(); ++i) {
            tmp_shape[i] = axes[perm[i]];
        }

        const DType* src_raw_data = cpu_data();
        Tensor<DType> result({broadcast_num, _size}, _backend);

        for(uint32_t i=0; i<broadcast_num; ++i) {
            for(uint32_t j=0; j<_size; ++j) {
                result.set_data_at({i,j}, src_raw_data[j]);
            }
        }

        result.Reshape(tmp_shape);
        result.Permute(perm);

        if(_backend==GPU)
            result.Cpu2Gpu();

        return result;
    }

    Tensor<DType> Var(std::vector<int32_t> axes, uint32_t correction=1) const {
        /* TODO validate axes */

        std::vector<uint32_t> perm;
        std::vector<uint32_t> reshape(2, 1);
        std::vector<uint32_t> result_shape;
        result_shape.reserve(_shape.size()-axes.size());

        std::vector<uint32_t> reduced_axes(axes.size());
        std::vector<uint32_t> reserved_axes;
        reserved_axes.reserve(_shape.size()-axes.size());

        /* convert axes into positive */
        for(uint32_t i=0; i<axes.size(); ++i) {
            reduced_axes[i] = CannonicalAxisIndex(axes[i]);
        } //reduced_axes=(0,2)

        for(uint32_t i=0; i<_shape.size(); ++i) {
            auto it = std::find(reduced_axes.begin(), reduced_axes.end(), i);
            if (it == reduced_axes.end()) {
                result_shape.emplace_back(shape(i));
                reserved_axes.emplace_back(i);
            }
        } // result_shape=(1,4); reserved_axes=(1,3)

        // perm=(1,3,0,2)
        perm.insert(perm.end(),reserved_axes.begin(),reserved_axes.end());
        perm.insert(perm.end(),reduced_axes.begin(),reduced_axes.end());

        for(uint32_t i=0; i<reserved_axes.size(); ++i) {
            reshape[0] *= _shape[reserved_axes[i]];
        } //reshape=(4,1)

        for(uint32_t i=0; i<reduced_axes.size(); ++i) {
            reshape[1] *= _shape[reduced_axes[i]];
        } // reshape=(4,6)

        Tensor<DType> tmp(_shape);
        Tensor<DType> mean = Mean(axes);
        Tensor<DType> result({reshape[0]});

        tmp.CopyFrom(*this);
        tmp.Permute(perm).Reshape(reshape); // tmp.shape=(6,4)
        const DType* mean_raw_val = mean.cpu_data();
        DType* result_raw_val = result.mutable_cpu_data();

        // tmp calc mean
        for(uint32_t i=0; i<reshape[0]; ++i) {
            DType sum = static_cast<DType>(0.);
            for(uint32_t j=0; j<reshape[1]; ++j) {
                sum += pow(tmp.data_at({i,j})-mean_raw_val[i], 2);
            }

            result_raw_val[i] = sum/static_cast<DType>(reshape[1]-correction);
        }

        result.Reshape(result_shape);

        return result;
    }

    /* 
       calc mean along axes ,
       dim of current tensor cannot beyond 4

       suppose tensor.shape=(2,1,3,4)-->(24)
       axes=(0,-2)
       result_shape=(1,4)
    */
    Tensor<DType> Mean(std::vector<int32_t> axes) const {
        /* TODO validate axes */

        std::vector<uint32_t> perm;
        std::vector<uint32_t> reshape(2, 1);
        std::vector<uint32_t> result_shape;
        result_shape.reserve(_shape.size()-axes.size());

        std::vector<uint32_t> reduced_axes(axes.size());
        std::vector<uint32_t> reserved_axes;
        reserved_axes.reserve(_shape.size()-axes.size());

        /* convert axes into positive */
        for(uint32_t i=0; i<axes.size(); ++i) {
            reduced_axes[i] = CannonicalAxisIndex(axes[i]);
        } //reduced_axes=(0,2)

        for(uint32_t i=0; i<_shape.size(); ++i) {
            auto it = std::find(reduced_axes.begin(), reduced_axes.end(), i);
            if (it == reduced_axes.end()) {
                result_shape.emplace_back(shape(i));
                reserved_axes.emplace_back(i);
            }
        } // result_shape=(1,4); reserved_axes=(1,3)

        // perm=(1,3,0,2)
        perm.insert(perm.end(),reserved_axes.begin(),reserved_axes.end());
        perm.insert(perm.end(),reduced_axes.begin(),reduced_axes.end());

        for(uint32_t i=0; i<reserved_axes.size(); ++i) {
            reshape[0] *= _shape[reserved_axes[i]];
        } //reshape=(4,1)

        for(uint32_t i=0; i<reduced_axes.size(); ++i) {
            reshape[1] *= _shape[reduced_axes[i]];
        } // reshape=(4,6)

        Tensor<DType> tmp(_shape);
        Tensor<DType> result({reshape[0]});

        tmp.CopyFrom(*this);
        tmp.Permute(perm).Reshape(reshape); // tmp.shape=(6,4)
        DType* result_raw_val = result.mutable_cpu_data();

        // tmp calc mean
        for(uint32_t i=0; i<reshape[0]; ++i) {
            DType sum = static_cast<DType>(0.);
            for(uint32_t j=0; j<reshape[1]; ++j) {
                sum += tmp.data_at({i,j});
            }

            result_raw_val[i] = sum/static_cast<DType>(reshape[1]);
        }

        result.Reshape(result_shape);
        return result;
    }

    /*  */
    void CopyFrom(const Tensor<DType> &src, bool reshape=false) {

        if(src.size()!=_size) {
            Warning("copy size does not match");
            exit(1);
        }

        //__cpy_raw_data(src.cpu_data(), src.size()*sizeof(DType));
        set_cpu_data(src.cpu_data());

        if(_backend==GPU) 
            Cpu2Gpu();

        if(_shape!=src.shape() && reshape) {
            ReshapeLike(src);
        }
    }

    /*  */
    inline std::string shape_string() const {
        std::ostringstream stream;

        stream << "(";
        for(auto &item: _shape) {
            stream << item << ",";
        }
        stream << ")"<< "-->(" <<  _size << ")"; 
        return stream.str();
    }

    /*  */
    inline const std::vector<uint32_t>& shape() const {
        return _shape;
    }

    /*  */
    inline uint32_t shape(int32_t index) const{
        return _shape[CannonicalAxisIndex(index)];
    }

    /*  */
    inline uint32_t size() const {return _size;}

    /* only positive start index and end index */
    inline uint32_t count(uint32_t sidx, uint32_t eidx) {
        auto num_axes = axes_num();
        if(!(CHECK_LE(sidx, eidx) && 
             CHECK_LE(sidx, num_axes) && 
             CHECK_LE(eidx, num_axes))) {

            std::ostringstream stream;
            stream << std::to_string(sidx) << " or " << 
                std::to_string(eidx) <<" is out of range";
            Warning(stream.str());
        }

        uint32_t count = 1ul;
        for(uint32_t i = sidx; i < eidx; ++i) {
            count *= shape(i);
        }

        return count;
    }

    /*  */
    inline uint32_t count(uint32_t sidx) {
        return count(sidx, axes_num());
    }

    /*  */
    inline uint32_t axes_num() const {
        return _shape.size();
    }

    /*  */
    inline uint32_t LegacyShape(int32_t index) const {
        int32_t num_axes = static_cast<int32_t>(axes_num());

        if(!(CHECK_GE(index, -_max_dim) &&
             CHECK_LT(index, _max_dim))) {
            std::ostringstream stream;
            stream << std::to_string(index) << " is out of range" ;
            Warning(stream.str());
        }

        if(CHECK_GE(index, num_axes) || CHECK_LT(index, -num_axes)){
            std::ostringstream stream;
            stream << std::to_string(index) << " is out of range"; 
            Warning(stream.str());
            return 0;
        }

        return shape(index);
    }

    inline uint32_t batch_size() {return shape(0);}
    inline uint32_t channel() {return shape(1);}
    inline uint32_t width() {return shape(2);}
    inline uint32_t height() {return shape(3);}

    /* transform a negative axis index to positive index */
    inline uint32_t CannonicalAxisIndex(int32_t axis_index) const {
        int32_t num_axes = static_cast<int32_t>(axes_num());

        if(!(CHECK_GE(axis_index, -num_axes) || CHECK_LE(axis_index, num_axes))){
            std::ostringstream stream;
            stream << std::to_string(axis_index) << " is out of range"; 
            Warning(stream.str());
        }

        if(axis_index < 0)
            return num_axes + axis_index;

        return axis_index;
    }

    /* shape=(n, c, h, w)*/
    inline uint32_t offset(uint32_t n, uint32_t c, uint32_t h, uint32_t w) const {
        if(!(CHECK_LE(n, batch_size()) && 
             CHECK_LE(n, channel())    && 
             CHECK_LE(n, width())      && 
             CHECK_LE(n, height()) )) {

            std::ostringstream stream;
            stream << std::to_string(n) << 
                " or " << std::to_string(c) <<
                " or " << std::to_string(h) <<
                " or " << std::to_string(w) <<
                "out of range";
            Warning(stream.str());
        }

        return ((n*channel()+c)*height()+h)*width()+w;
    }

    inline uint32_t offset(std::vector<uint32_t> indices) const {
        if(!CHECK_LE(indices.size(), axes_num())) {
            std::ostringstream stream;
            stream << std::to_string(indices.size()) << "out of range";
            Warning(stream.str());
        }

        uint32_t offset = 0;
        for(uint32_t i = 0; i<axes_num(); ++i) {
            offset *= shape(i);

            if(i < indices.size()) {
                if(CHECK_GT(indices[i], shape(i))) {
                    std::ostringstream stream;
                    stream << std::to_string(indices[i]) << "out of range for ";
                    Warning(stream.str());
                    exit(1);
                }
                offset += indices[i];
            }
        }

        return offset;
    }

    void Gpu2Cpu() {
        if(_backend==CPU) {
            Warning("backend is cpu, do not need to sync to cpu");
            return;
        }

        _data->sync_gpu2cpu();
    }

    void Cpu2Gpu() {
        if(_backend==CPU) {
            Warning("backend is cpu, do not need to sync to cpu");
            return;
        }

        _data->sync_cpu2gpu();
    }

    Tensor<DType> ElementWiseAdd(const Tensor<DType> &other) const {
        if(_shape!=other.shape()) {
            Warning("element wise mul shape does not match");
            exit(1);
        }

        Tensor<DType> result(_shape, _backend);

        if(_backend==CPU) {
            DType* result_raw_data = static_cast<DType*>(result.mutable_cpu_data());
            const DType* raw_data = cpu_data();
            const DType* other_raw_data = other.cpu_data();

            for(uint32_t i=0; i<_size; ++i)
                result_raw_data[i] = raw_data[i]+other_raw_data[i];
        }

        if(_backend==GPU) {
            DType* result_raw_data = static_cast<DType*>(result.mutable_gpu_data());
            const DType* raw_data = static_cast<const DType*>(gpu_data());
            const DType* other_raw_data = static_cast<const DType*>(other.gpu_data());
            vec_add(raw_data, other_raw_data, result_raw_data, _size);
            result.Gpu2Cpu();
        }

        return result;
    }

    Tensor<DType> ElementWiseMinus(const Tensor<DType> &other) const {
        if(_shape!=other.shape()) {
            Warning("element wise mul shape does not match");
            exit(1);
        }

        Tensor<DType> result(_shape);

        DType* result_raw_data = result.mutable_cpu_data();
        const DType* raw_data = cpu_data();
        const DType* other_raw_data = other.cpu_data();

        for(uint32_t i=0; i<_size; ++i) {
            result_raw_data[i] = raw_data[i]-other_raw_data[i];
        }

        return result;
    }

    Tensor<DType> ElementWiseMul(const Tensor<DType> &other) const {
        if(_shape!=other.shape()) {
            Warning("element wise mul shape does not match");
            exit(1);
        }

        Tensor<DType> result(_shape);

        DType* result_raw_data = result.mutable_cpu_data();
        const DType* raw_data = cpu_data();
        const DType* other_raw_data = other.cpu_data();

        for(uint32_t i=0; i<_size; ++i) {
            result_raw_data[i] = raw_data[i]*other_raw_data[i];
        }

        return result;
    }

    Tensor<DType> ElementWiseDiv(const Tensor<DType> &other, 
                                 float eps=1e-5) const {
        if(_shape!=other.shape()) {
            Warning("element wise mul shape does not match");
            exit(1);
        }

        Tensor<DType> result(_shape);

        DType* result_raw_data = result.mutable_cpu_data();
        const DType* raw_data = cpu_data();
        const DType* other_raw_data = other.cpu_data();

        for(uint32_t i=0; i<_size; ++i) {
            DType denominator = other_raw_data[i];
            if(denominator == static_cast<DType>(0.))
                denominator = static_cast<DType>(eps);

            result_raw_data[i] = raw_data[i]/denominator;
        }

        return result;
    }

    Tensor<DType>& operator=(const Tensor<DType> &other) { 

        _shape = other.shape();
        _size = other.size();
        _capacity = _size;
        _shape_data.reset(new SyncMem(_shape.size() * sizeof(uint32_t)));
        _data.reset(new SyncMem(_capacity*sizeof(DType)));

        CopyFrom(other);

        return *this;
    }

    Tensor<DType> operator*(const Tensor<DType> &other) const { 

        if(other.shape() != _shape) {
            if(other.axes_num() <= axes_num())
                return ElementWiseMul(other.BroadCast(_shape));
            else
                return other.ElementWiseMul(BroadCast(other.shape()));
        }

        return ElementWiseMul(other);
    }

    Tensor<DType> operator/(const Tensor<DType> &other) const {

        if(other.shape() != _shape) {
            if(other.axes_num() <= axes_num())
                return ElementWiseDiv(other.BroadCast(_shape));
            else
                return other.ElementWiseDiv(BroadCast(other.shape()));
        }

        return ElementWiseDiv(other);
    }

    Tensor<DType> operator+(const Tensor<DType> &other) const { 
        if(other.shape() != _shape) {
            if(other.axes_num() <= axes_num())
                return ElementWiseAdd(other.BroadCast(_shape));
            else
                return other.ElementWiseAdd(BroadCast(other.shape()));
        }

        return ElementWiseAdd(other);
    }

    Tensor<DType> operator-(const Tensor<DType> &other) const { 
        if(other.shape() != _shape) {
            if(other.axes_num() <= axes_num())
                return ElementWiseMinus(other.BroadCast(_shape));
            else
                return other.ElementWiseMinus(BroadCast(other.shape()));
        }

        return ElementWiseMinus(other);

    }

    Tensor<DType> Sqrt() const {
        Tensor<DType> result(_shape);

        DType* result_raw_data = result.mutable_cpu_data();
        const DType* raw_data = cpu_data();

        for(uint32_t i=0; i<_size; ++i) {
            DType raw = raw_data[i];
            if(raw<static_cast<DType>(0.)){
                Warning("sqrt element is negative");
                exit(1);
            }

            result_raw_data[i] = pow(raw, 0.5);
        }

        return result;
    }

    /*  */
    const DType* cpu_data() const {
        if(_data==nullptr) {
            Warning("data is null");
            exit(1);
        }

        return (const DType*)_data->cpu_data();
    }

    /*  */
    const DType* gpu_data() const {
        if(_data==nullptr) {
            Warning("data is null");
            exit(1);
        }

        return (const DType*)_data->gpu_data();
    }

    /*  */
    inline DType data_at(uint32_t n, uint32_t c=0, 
                         uint32_t h=0, uint32_t w=0) const {
        return cpu_data()[offset(n,c,h,w)];
    }

    /*  */
    inline DType data_at(const std::vector<uint32_t> indices) const {
        return cpu_data()[offset(indices)];
    }

    /*  */
    inline void set_data_at(const std::vector<uint32_t> indices, DType value) {
        DType* tmp_data = mutable_cpu_data();
        tmp_data[offset(indices)] = value;
    }


    /*  */
    inline const std::shared_ptr<SyncMem>& data(){
        //if(_data==nullptr) {
        //}

        return _data;
    }

    DType* mutable_gpu_data() {
        if(_data==nullptr) {
            /*TODO*/
        }

        return static_cast<DType*>(_data->mutable_gpu_data());
    }

    /*  */
    DType* mutable_cpu_data() {
        if(_data==nullptr) {
            /*TODO*/
        }

        return static_cast<DType*>(_data->mutable_cpu_data());
    }
    //void FromFlat(const TensorFlatT* flat, bool reshape=true);

    void set_gpu_data(const DType* data) {
        if(_data==nullptr) {
            Warning("data is nullptr");
            exit(1);
        }

        // Make sure CPU and GPU sizes remain equal
        uint32_t size = _size* sizeof(DType);
        if (_data->size() != size) {
          _data.reset(new SyncMem(size));
        }

        /* deep copy in syncmem for safe */
        _data->set_gpu_data(static_cast<const void*>(data), _size*(sizeof(DType)));
    }

    /*  */
    void set_cpu_data(const DType* data) {
        if(_data==nullptr) {
            Warning("data is nullptr");
            exit(1);
        }

        // Make sure CPU and GPU sizes remain equal
        uint32_t size = _size* sizeof(DType);
        if (_data->size() != size) {
          _data.reset(new SyncMem(size));
        }

        /* deep copy in syncmem for safe */
        _data->set_cpu_data(static_cast<const void*>(data), _size*(sizeof(DType)));
    }

    /*  */
    void scale_data(DType scale_factor);

    /*  */
    void ShareData(const Tensor& other);
    //bool ShapeEq(const Tensor& o);

    /* suppose this is for 4 dims mm */
    Tensor<DType> mm(const Tensor<DType> &other) const {
        std::vector<uint32_t> b_shape = other.shape();

        if(b_shape.size() < 2){
            Warning("dim of input tensor is less than 2");
            exit(1);
        }
        uint32_t b_row = other.shape(-2);
        uint32_t b_col = other.shape(-1);
        uint32_t a_row = shape(-2);
        uint32_t a_col = shape(-1);

        if(b_row!=a_col) {
            Warning("input dim does not match");
            exit(1);
        }

        std::vector<uint32_t> result_shape = b_shape;
        //result_shape.reserve(b_shape.size());

        uint32_t tmp_dim = 1ul;
        uint32_t dim = 0;
        for(; dim<b_shape.size()-2; ++dim) {
            //result_shape.emplace_back(b_shape[i]);
            tmp_dim *= b_shape[dim];
        }

        result_shape[dim] = a_row;
        result_shape[dim+1] = b_col;

        Tensor<DType> result({tmp_dim, a_row, b_col});
        const DType* A_raw_data = cpu_data();
        const DType* B_raw_data = other.cpu_data();
        //DType* result_data = result.mutable_cpu_data();

        /* 
           (bs, a_row, a_col) @ (bs, b_row, b_col) ==> (bs, a_row, b_col)
           a_col == b_row
           tedious!!!
         */
        for(uint32_t bs=0; bs<tmp_dim; ++bs) {
            /* (bs, i,j)*(bs, j,k) --> (bs, i,k)*/
            for(uint32_t i=0; i<a_row; ++i) {
                for(uint32_t k=0; k<b_col; ++k) {

                    DType tmp = static_cast<DType>(0);
                    for(uint32_t j=0; j<a_col; ++j) {
                        //((n*channel()+c)*height()+h)*width()+w;
                        uint32_t A_offset = (bs*a_row+i)*a_col+j;
                        uint32_t B_offset = (bs*b_row+j)*b_col+k;
                        tmp += A_raw_data[A_offset]*B_raw_data[B_offset];
                    } // j

                    result.set_data_at({bs,i,k}, tmp);
                } // k
            } // i
        } // bs

        result.Reshape(result_shape);
        return result;
    }

private:
    Tensor<DType>& __permute2(const std::vector<uint32_t>& perm) {
        // TODO: validate perm

        std::vector<uint32_t> result_shape = {_shape[perm[0]], 
                                              _shape[perm[1]]};
        Tensor<DType> tmp_t(result_shape);

        DType* tmp_data = tmp_t.mutable_cpu_data();
        DType* raw_data = mutable_cpu_data();

        std::vector<uint32_t> tmp(perm.size());

        uint32_t dim0 = _shape[0];
        uint32_t dim1 = _shape[1];

        for(;tmp[0]<dim0; ++tmp[0]) {
            tmp[1] = 0;
            for(;tmp[1]<dim1; ++tmp[1]) {

                auto offset1 = offset({tmp[0], tmp[1]});
                auto offset2 = tmp_t.offset({tmp[perm[0]], tmp[perm[1]]});
                tmp_data[offset2] = raw_data[offset1];
            } // 1
        } // 0

        CopyFrom(tmp_t, true);
        return *this;
    }

    Tensor<DType>& __permute3(const std::vector<uint32_t>& perm) {
        // TODO: validate perm

        std::vector<uint32_t> result_shape = {_shape[perm[0]], 
                                              _shape[perm[1]], 
                                              _shape[perm[2]]};
        Tensor<DType> tmp_t(result_shape);

        DType* tmp_data = tmp_t.mutable_cpu_data();
        DType* raw_data = mutable_cpu_data();

        std::vector<uint32_t> tmp(perm.size());

        uint32_t dim0 = _shape[0];
        uint32_t dim1 = _shape[1];
        uint32_t dim2 = _shape[2];

        for(;tmp[0]<dim0; ++tmp[0]) {
            tmp[1] = 0; tmp[2] = 0;
            for(;tmp[1]<dim1; ++tmp[1]) {
                tmp[2] = 0; 
                for(;tmp[2]<dim2; ++tmp[2]) {

                    auto offset1 = offset({tmp[0], tmp[1], tmp[2]});
                    auto offset2 = tmp_t.offset({tmp[perm[0]], 
                                                 tmp[perm[1]], 
                                                 tmp[perm[2]]});

                    tmp_data[offset2] = raw_data[offset1];
                } //2
            } // 1
        } // 0

        CopyFrom(tmp_t, true);
        return *this;
    }

    Tensor<DType>& __permute4(const std::vector<uint32_t>& perm) {
        // TODO: validate perm

        std::vector<uint32_t> result_shape = {_shape[perm[0]], _shape[perm[1]], 
                                              _shape[perm[2]], _shape[perm[3]]};
        Tensor<DType> tmp_t(result_shape);

        DType* tmp_data = tmp_t.mutable_cpu_data();
        DType* raw_data = mutable_cpu_data();

        std::vector<uint32_t> tmp(perm.size());

        uint32_t dim0 = _shape[0];
        uint32_t dim1 = _shape[1];
        uint32_t dim2 = _shape[2];
        uint32_t dim3 = _shape[3];

        for(;tmp[0]<dim0; ++tmp[0]) {
            tmp[1] = 0; tmp[2] = 0; tmp[3] = 0;
            for(;tmp[1]<dim1; ++tmp[1]) {
                tmp[2] = 0; tmp[3] = 0;
                for(;tmp[2]<dim2; ++tmp[2]) {
                    tmp[3] = 0;
                    for(;tmp[3]<dim3; ++tmp[3]) {

                        auto offset1 = offset({tmp[0], tmp[1], tmp[2], tmp[3]});
                        auto offset2 = tmp_t.offset({tmp[perm[0]], tmp[perm[1]], 
                                               tmp[perm[2]], tmp[perm[3]]});

                        tmp_data[offset2] = raw_data[offset1];
                    } // 3
                } //2
            } // 1
        } // 0

        CopyFrom(tmp_t, true);
        return *this;
    }

    /*
    void __cpy_raw_data(const DType* src, uint32_t len) {
        DType* raw_data = mutable_cpu_data();
        uint32_t cpy_size = _size*sizeof(DType);

        if(len!=cpy_size) {
            Warning("data size does not match");
            exit(1);
        }

        if(raw_data != src) {
            std::memcpy(raw_data, src, cpy_size);
        }
    }
    */

protected:
    std::shared_ptr<SyncMem> _data; // tensor data info
    std::shared_ptr<SyncMem> _shape_data;
    std::vector<uint32_t> _shape; // only shape info
    uint32_t _size; // the same as vector
    uint32_t _capacity; // the same as vector
    const uint32_t _max_dim;
    BackEnd _backend;

    //DISABLE_COPY_AND_ASSIGN(Tensor);

}; // class Tensor
}; // namespace geefer
#endif

