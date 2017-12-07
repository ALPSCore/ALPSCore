//
// Created by iskakoff on 26/10/17.
//

#ifndef GF2_TENSOR_H
#define GF2_TENSOR_H


#include <vector>
#include <array>
#include <iostream>
#include <type_traits>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "../type_traits.h"
#include "DataView.h"



namespace alps {
  namespace gf {
    namespace detail {
      template<typename T, typename St>
      struct is_storage {
        static constexpr bool value = std::is_same < St, DataStorage <T> >::value || std::is_same < St, DataView <T> >::value;
      };
      /**
       * Base Tensor Class
       * @tparam T - datatype, should be scalar
       * @tparam D - dimension of tensor
       * @tparam C - type of the container, either DataStorage or DataView
       */
      template<typename T, int D, typename C>
      class TensorBase;

      /**
       * Definition of Tensor with storage
       */
      template<typename T, int D>
      using Tensor = TensorBase < T, D, DataStorage < T > >;
      /**
       * Definition of Tensor as view of existent data array
       */
      template<typename T, int D>
      using TensorView = TensorBase < T, D, DataView < T > >;


      template<typename X, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
      using MatrixMap =  Eigen::Map < Eigen::Matrix < X, Rows, Cols, Eigen::RowMajor > >;

      /**
       * @brief Tensor class for raw data storage and performing the basic arithmetic operations
       *
       * @author iskakoff
       */
      template<typename T, int D, typename C>
      class TensorBase {
        typedef T prec;
        static int constexpr dim = D;
        typedef DataView < T > viewType;
        typedef DataStorage < T > storageType;
        typedef TensorBase < T, D, C > tType;
        typedef Tensor < T, D > tensorType;
        typedef TensorView < T, D > tensorViewType;
        template<typename St>
        using   genericTensor = TensorBase < T, D, St >;

      public:

        /**
         *
         * @param container
         * @param sizes
         */
        TensorBase(C &&container, std::array < size_t, D > sizes) : _data(container), _sizes(sizes) {
          static_assert(is_storage< T, C>::value, "Should be either DataStorage or DataView type");
          fill_acc_sizes();
        }

        TensorBase(C &container, std::array < size_t, D > sizes) : _data(container), _sizes(sizes) {
          static_assert(is_storage< T, C>::value, "Should be either DataStorage or DataView type");
          fill_acc_sizes();
        }

        /**
         * Create tensor from the existent data. All operation will be performed on the data stored in <data> parameter
         * if DataView storage is used. In case of DataStorage storage all data will be copied into raw vector.
         *
         * @param data  - pointer to the raw data buffer
         * @param sizes - array with sizes for each dimension
         */
        TensorBase(T *data, std::array < size_t, D > sizes) : _data(viewType(data, size(sizes))), _sizes(sizes) {
          fill_acc_sizes();
        }

        /**
         * Create empty Tensor with provided sizes for each dimensions
         *
         * @tparam X type of storage to be created. Should always be DataStorage
         * @param sizes - array of data dimensions
         */
        template<typename X = C>
        TensorBase(typename std::enable_if < std::is_same < X, DataStorage < T > >::value, const std::array < size_t, D > & >::type sizes) : _data(size(sizes)), _sizes(sizes) {
          fill_acc_sizes();
        }

        explicit TensorBase(size_t size) : _data(size), _sizes{{size}} {
          static_assert(1 == D, "Wrong dimension");
          fill_acc_sizes();
        }

        /// copy constructor
        TensorBase(const tensorType& rhs) : _data(rhs.data()), _sizes(rhs.sizes()), _acc_sizes(rhs.acc_sizes()) {}
        /// move constructor
        TensorBase(tensorType &&rhs) : _data(rhs.data()), _sizes(rhs.sizes()), _acc_sizes(rhs.acc_sizes()) {}
        /// copy constructor
        TensorBase(const tensorViewType &rhs) : _data(rhs.data()), _sizes(rhs.sizes()), _acc_sizes(rhs.acc_sizes()) {}
        /// move constructor
        TensorBase(tensorViewType &&rhs) : _data(rhs.data()), _sizes(rhs.sizes()), _acc_sizes(rhs.acc_sizes()) {}


        /// Copy assignment
        TensorBase < T, D, C > &operator=(const TensorBase < T, D, C > &rhs) {
          _data = rhs._data;
          _sizes = rhs._sizes;
          _acc_sizes = rhs._acc_sizes;
          return *this;
        }

        /// Move assignment
        TensorBase < T, D, C > &operator=(TensorBase < T, D, C > &&rhs) noexcept {
          _data = std::move(rhs._data);
          _sizes = std::move(rhs._sizes);
          _acc_sizes = std::move(rhs._acc_sizes);
          return *this;
        }

        /**
         * Get data point for the specific set of indices
         * @tparam IndexTypes - types of tail indices
         * @param t1          - head index
         * @param indices     - tail indices
         * @return value of tensor at the (t1, indices...) point
         */
        template<typename ...IndexTypes>
        T operator()(typename std::enable_if < sizeof...(IndexTypes) == D - 1, size_t >::type t1, IndexTypes ... indices) const {
          return _data.data(index(std::integral_constant < int, 0 >(), t1, indices...));
        }

        /**
         * Get reference to the data point at the (t1, indices...) point
         */
        template<typename ...IndexTypes>
        T &operator()(typename std::enable_if < sizeof...(IndexTypes) == D - 1, size_t >::type t1, IndexTypes ... indices) {
          return _data.data(index(std::integral_constant < int, 0 >(), t1, indices...));
        }


        /**
         * Get slice of the Tensor for the specific leading indices set.
         * This method creates a Tensor view object of the smaller dimension
         * for the part of the current Tensor object for the specific set of indices.
         *
         * @tparam IndexTypes - types of tail indices
         * @param t1          - first leading index
         * @param indices     - rest indices
         * @return slice for the specific leading indices
         */
        template<typename ...IndexTypes>
        TensorView < T, D - (sizeof...(IndexTypes)) - 1 > operator()(typename std::enable_if < (sizeof...(IndexTypes) < D - 1), size_t >::type t1, IndexTypes ... indices) {
          std::array < size_t, D - (sizeof...(IndexTypes)) - 1 > sizes;
          size_t s = 1;
          for (int i = 0; i < sizes.size(); ++i) {
            sizes[i] = _sizes[i + sizeof...(IndexTypes) + 1];
            s *= sizes[i];
          }
          return TensorView < T, D - (sizeof...(IndexTypes)) - 1 >(viewType(_data, s, (index(std::integral_constant < int, 0 >(), t1, indices...))), sizes);
        }

        /*
         * Basic arithmetic operations
         */
        /**
         * @tparam S     - type of scalar multiplier (If tensor is real, S should be real, If tensor is complex S can be any scalar type.)
         * @param scalar - scalar factor
         * @return New tensor equal to the current tensor multiplied by scalar
         */
        template<typename S>
        typename std::enable_if < !std::is_same < S, tensorType >::value && std::is_scalar < S >::value, tensorType >::type operator*(S scalar) {
          tensorType x(*this);
          x *= T(scalar);
          return std::move(x);
        };

        /**
         * Scaling by complex scalar. Method returns new complex Tensor that is the result of scaling the current real tensor object by complex scalar.
         * This method checks that the type of the current Tensor is not complex.
         *
         * @tparam S     - type of scalar multiplier (If tensor is real, S should be real, If tensor is complex S can be any scalar type.)
         * @param scalar - scalar factor
         * @return New tensor equal to the current tensor multiplied by scalar
         */
        template<typename S, int M = D>
        typename std::enable_if < !std::is_same < S, tensorType >::value && (is_complex < S >::value && !is_complex < T >::value), Tensor < S, M > >::type operator*(S scalar) {
          Tensor < S, M > x(DataStorage < S >(_data.size()), this->_sizes);
          Eigen::Map < Eigen::Matrix < T, 1, Eigen::Dynamic, Eigen::RowMajor > > M1(&_data.data(0), _data.size());
          Eigen::Map < Eigen::Matrix < S, 1, Eigen::Dynamic, Eigen::RowMajor > > M2(&x.data().data(0), _data.size());
          M2 = M1;
          x *= scalar;
          return std::move(x);
        };

        /**
         * Two tensors multiplication. We need to decide whether it should be element-wise or more sofisticated tensor contraction.
         *
         * @tparam S     - type of the right hand side tensor
         * @param tensor - right hand side tensor
         * @return result of two tensor multiplication
         */
        template<typename S>
        typename std::enable_if < std::is_same < S, tensorType >::value, tensorType >::type operator*(S tensor) {
          throw std::runtime_error("Function is not implemented yet.");
        };

        /**
         * Inplace tensor scaling
         */
        template<typename S>
        typename std::enable_if < !std::is_same < S, tensorType >::value, tType & >::type operator*=(S scalar) {
          Eigen::Map < Eigen::Matrix < T, 1, Eigen::Dynamic > > M(&_data.data(0), _data.size());
          M *= T(scalar);
          return *this;
        };

        /**
         * Inplace tensor multiplication
         */
        template<typename S>
        typename std::enable_if < std::is_same < S, tensorType >::value, tensorType & >::type operator*=(S tensor) {
          throw std::runtime_error("Function is not implemented yet.");
        };

        /**
         * Tensor inversed scaling
         */
        template<typename S>
        typename std::enable_if < !std::is_same < S, tensorType >::value && std::is_scalar < S >::value, tensorType >::type operator/(S scalar) {
          tensorType x(*this);
          x /= T(scalar);
          return std::move(x);
        };

        /**
         * Real value Tensor division by complex scalar. Method will create complex Tensor that equals to current real tensor divided by complex scalar.
         */
        template<typename S, int M = D>
        typename std::enable_if < !std::is_same < S, tensorType >::value && (is_complex < S >::value && !is_complex < T >::value), Tensor < S, M > >::type operator/(S scalar) {
          Tensor < S, M > x(DataStorage < S >(_data.size()), this->_sizes);
          Eigen::Map < Eigen::Matrix < T, 1, Eigen::Dynamic, Eigen::RowMajor > > M1(&_data.data(0), _data.size());
          Eigen::Map < Eigen::Matrix < S, 1, Eigen::Dynamic, Eigen::RowMajor > > M2(&x.data().data(0), _data.size());
          M2 = M1;
          x /= scalar;
          return std::move(x);
        };

        /**
         * Inplace division
         */
        template<typename S>
        typename std::enable_if < !std::is_same < S, tensorType >::value, tType & >::type operator/=(S scalar) {
          Eigen::Map < Eigen::Matrix < T, 1, Eigen::Dynamic > > M(&_data.data(0), _data.size());
          M /= T(scalar);
          return *this;
        };

        /**
         * @return raw data buffer size
         */
        size_t size() const {
          return _data.size();
        }

        /**
         * For 2D square Tensor compute inverse Tensor.
         * @return inversed Tensor
         */
        template<int M = D>
        typename std::enable_if < M == 2, tensorType >::type
        inverse() {
          if (_sizes[0] != _sizes[1]) {
            throw std::invalid_argument("Can not do inversion of the non-square matrix.");
          }
          tensorType x(*this);
          Eigen::Map < Eigen::Matrix < T, Eigen::Dynamic, Eigen::Dynamic > > Mt(&(x.data().data(0)), _sizes[0], _sizes[1]);
          Mt = Mt.inverse();
          return std::move(x);
        };

        /**
         * Compute sum of two tensors if either left-hand-side tensor is complex or lhs and rhs tensors are of the same type.
         *
         * @tparam S  - data type of rhs tensor
         * @tparam Ct - storage type of rhs tensor
         * @param  y  - rhs tensor
         * @return new tensor object that equals to sum of current tensor and rhs tensor
         */
        template<typename S, typename Ct>
        typename std::enable_if <
          std::is_same < S, T >::value || std::is_same < T, std::complex < double>>::value || std::is_same < T, std::complex < float>>::value, tensorType >::type
        operator+(const TensorBase < S, D, Ct > &y) {
          tensorType x(*this);
          x += y;
          return std::move(x);
        };

        /**
         * Compute sum of two tensors if right-hand-side tensor is complex and rhs and lhs tensor are of different type.
         *
         * @tparam S  - data type of rhs tensor
         * @tparam Ct - storage type of rhs tensor
         * @param  y  - rhs tensor
         * @return new tensor object that equals to sum of current tensor and rhs tensor
         */
        template<typename S, typename Ct>
        typename std::enable_if <
          !std::is_same < S, T >::value && (std::is_same < S, std::complex < double>>::value || std::is_same < S, std::complex < float>>::value), Tensor < S, D > >::type
        operator+(const TensorBase < S, D, Ct > &y) {
          Tensor < S, D > x(y);
          x += *this;
          return std::move(x);
        };

        /**
         * Compute difference of two tensors of a same type
         */
        template<typename S>
        typename std::enable_if < std::is_same < S, tensorType >::value || std::is_same < S, tensorViewType >::value, tensorType >::type operator-(const S &y) {
          tensorType x(*this);
          x -= y;
          return std::move(x);
        };

        /**
         * Compute difference of two tensors of a different type
         */
        template<typename S, typename Ct>
        typename std::enable_if <
          !std::is_same < S, T >::value && (std::is_same < S, std::complex < double>>::value || std::is_same < S, std::complex < float>>::value), Tensor < S, D > >::type
        operator-(const TensorBase < S, D, Ct > &y) {
          Tensor < S, D > x(y);
          x -= *this;
          return std::move(x);
        };

        /**
         * Inplace addition
         */
        template<typename S, typename Ct>
        typename std::enable_if <
          std::is_same < S, T >::value || std::is_same < T, std::complex < double>>::value || std::is_same < T, std::complex < float>>::value, tType & >::type
        operator+=(const TensorBase < S, D, Ct > &y) {
          MatrixMap < T, 1, Eigen::Dynamic > M1(&_data.data(0), _data.size());
          Eigen::Map < const Eigen::Matrix < S, 1, Eigen::Dynamic, Eigen::RowMajor > > M2(&y.data().data(0), y.data().size());
          M1.noalias() += M2;
          return (*this);
        };

        /**
         * Inplace subtraction
         */
        template<typename S>
        typename std::enable_if < std::is_same < S, tensorType >::value || std::is_same < S, tensorViewType >::value, tType & >::type operator-=(const S &y) {
          MatrixMap < T, 1, Eigen::Dynamic > M1(&_data.data(0), _data.size());
          Eigen::Map < const Eigen::Matrix < T, 1, Eigen::Dynamic, Eigen::RowMajor > > M2(&y.data().data(0), y.data().size());
          M1.noalias() -= M2;
          return (*this);
        };

        /**
         * Compute a dot product of two 2D tensors
         */
        template<int M = D>
        typename std::enable_if < M == 2, tensorType >::type
        dot(const tensorType &y) {
          if (_sizes[0] != y._sizes[1] && _sizes[1] != y._sizes[0]) {
            throw std::invalid_argument("Can not do multiplication. Dimensions missmatches.");
          }
          tensorType x(*this);
          MatrixMap < T > M1(&x.data().data(0), _sizes[0], _sizes[1]);
          Eigen::Map < const Eigen::Matrix < T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > > M2(&y.data().data(0), _sizes[0], _sizes[1]);
          M1 *= M2;
          return std::move(x);
        };

        /**
         * @return Eigen matrix representation for 2D Tensor
         */
        template<int M = D>
        typename std::enable_if < M == 2, MatrixMap < T > >::type
        matrix() {
          return MatrixMap < T >(&data().data(0), _sizes[0], _sizes[1]);
        };



        /// sizes for each dimension
        const std::array < size_t, D > &sizes() const { return _sizes; };

        /// offset multipliers for each dimension
        const std::array < size_t, D > &acc_sizes() const { return _acc_sizes; };

        /// dimension of the tensor
        int dimension() const { return D; }

        /// const reference to the data storage
        const C &data() const { return _data; }

        /// reference to the data storage
        C &data() { return _data; }

      private:
        /// data storage object
        C _data;
        /// stored sizes for each dimensions
        std::array < size_t, D > _sizes;
        /// offset multiplier for each dimension
        std::array < size_t, D > _acc_sizes;

        /// compte size for the specific dimensions
        size_t size(const std::array < size_t, D > &sizes) {
          size_t res = 1;
          for (int i = 0; i < D; ++i) {
            res *= sizes[i];
          }
          return res;
        }

      public:

        // these methods should be private but they are public for the test purposes

        /// return index in the raw buffer for specified indices
        template<int M, typename Ti, typename ...Ts>
        size_t index(std::integral_constant < int, M > i, const Ti &t1, const Ts &...ts) const {
          return t1 * _acc_sizes[i.value] + index(std::integral_constant < int, M + 1 >(), std::forward < const Ts & >(ts)...);
        }

        /// return index in the raw buffer for the last index
        template<int M, typename Ti>
        size_t index(std::integral_constant < int, M > i, const Ti &t1) const {
          return t1 * _acc_sizes[i.value];
        }

      private:
        /**
         * compute offset multiplier for each dimension
         */
        void fill_acc_sizes() {
          int k = 1;
          for (size_t &i : _acc_sizes) {
            i = 1;
            for (int j = k; j < _sizes.size(); ++j) {
              i *= _sizes[j];
            }
            ++k;
          }
        }
      };

    }
  }
}

#endif //GF2_TENSOR_H
