/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef GF2_TENSOR_H
#define GF2_TENSOR_H


#include <vector>
#include <array>
#include <iostream>
#include <type_traits>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <alps/gf_new/type_traits.h>
#include <alps/gf_new/tensors/data_view.h>



namespace alps {
  namespace gf {
    namespace detail {
      template<typename T, typename St>
      struct is_storage {
        static constexpr bool value = std::is_same < St, data_storage <T> >::value || std::is_same < St, data_view <T> >::value;
      };
      /**
       * Base Tensor Class
       * @tparam T - datatype, should be scalar
       * @tparam D - dimension of tensor
       * @tparam C - type of the container, either DataStorage or DataView
       */
      template<typename T, int D, typename C>
      class tensor_base;

      /**
       * Definition of Tensor with storage
       */
      template<typename T, int D>
      using tensor = tensor_base < T, D, data_storage < T > >;
      /**
       * Definition of Tensor as view of existent data array
       */
      template<typename T, int D>
      using tensor_view = tensor_base < T, D, data_view < T > >;


      template<typename X, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
      using MatrixMap =  Eigen::Map < Eigen::Matrix < X, Rows, Cols, Eigen::RowMajor > >;

      /**
       * @brief Tensor class for raw data storage and performing the basic arithmetic operations
       *
       * @author iskakoff
       */
      template<typename T, int D, typename C>
      class tensor_base {
        // types definitions
        typedef T prec;
        typedef data_view < T > viewType;
        typedef data_storage < T > storageType;
        /// current Tensor type
        typedef tensor_base < T, D, C > tType;
        /// Tensor type with storage
        typedef tensor < T, D > tensorType;
        /// view Tensor type
        typedef tensor_view < T, D > tensorViewType;
        /// generic tensor type
        template<typename St>
        using   genericTensor = tensor_base < T, D, St >;

      private:
        // fields definitions
        /// tensor dimension
        static int constexpr dim = D;
        /// data storage object
        C data_;
        /// stored sizes for each dimensions
        std::array < size_t, D > sizes_;
        /// offset multiplier for each dimension
        std::array < size_t, D > acc_sizes_;

      public:

        /**
         *
         * @param container
         * @param sizes
         */
        tensor_base(C &&container, std::array < size_t, D > sizes) : data_(container), sizes_(sizes) {
          static_assert(is_storage< T, C>::value, "Should be either data_storage or data_view type");
          fill_acc_sizes();
        }

        tensor_base(C &container, std::array < size_t, D > sizes) : data_(container), sizes_(sizes) {
          static_assert(is_storage< T, C>::value, "Should be either data_storage or data_view type");
          fill_acc_sizes();
        }

        /**
         * Create tensor from the existent data. All operation will be performed on the data stored in <data> parameter
         * if DataView storage is used. In case of DataStorage storage all data will be copied into raw vector.
         *
         * @param data  - pointer to the raw data buffer
         * @param sizes - array with sizes for each dimension
         */
        tensor_base(T *data, std::array < size_t, D > sizes) : data_(viewType(data, size(sizes))), sizes_(sizes) {
          fill_acc_sizes();
        }

        /**
         * Create empty Tensor with provided sizes for each dimensions
         *
         * @tparam X type of storage to be created. Should always be DataStorage
         * @param sizes - array of data dimensions
         */
        template<typename X = C>
        tensor_base(typename std::enable_if < std::is_same < X, data_storage < T > >::value, const std::array < size_t, D > & >::type sizes) : data_(size(sizes)), sizes_(sizes) {
          fill_acc_sizes();
        }

        explicit tensor_base(size_t size) : data_(size), sizes_{{size}} {
          static_assert(1 == D, "Wrong dimension");
          fill_acc_sizes();
        }

        /// copy constructor
        tensor_base(const tensorType& rhs) : data_(rhs.data()), sizes_(rhs.sizes()), acc_sizes_(rhs.acc_sizes()) {}

        template<typename St = C>
        tensor_base(typename std::enable_if<std::is_same<St, viewType>::value, tensorType>::type& rhs) :
          data_(rhs.data()), sizes_(rhs.sizes()), acc_sizes_(rhs.acc_sizes()) {}
        /// move constructor
        tensor_base(tensorType &&rhs) : data_(rhs.data()), sizes_(rhs.sizes()), acc_sizes_(rhs.acc_sizes()) {}
        /// copy constructor
        tensor_base(const tensorViewType &rhs) : data_(rhs.data()), sizes_(rhs.sizes()), acc_sizes_(rhs.acc_sizes()) {}
        /// move constructor
        tensor_base(tensorViewType &&rhs) : data_(rhs.data()), sizes_(rhs.sizes()), acc_sizes_(rhs.acc_sizes()) {}


        /// Copy assignment
        tensor_base < T, D, C > &operator=(const tensor_base < T, D, C > &rhs) {
          data_ = rhs.data_;
          sizes_ = rhs.sizes_;
          acc_sizes_ = rhs.acc_sizes_;
          return *this;
        }

        /// Move assignment
        tensor_base < T, D, C > &operator=(tensor_base < T, D, C > &&rhs) noexcept {
          data_ = rhs.data_;
          sizes_ = rhs.sizes_;
          acc_sizes_ = rhs.acc_sizes_;
          return *this;
        }

        bool operator==(const tType& rhs) const {
          return std::equal(sizes_.begin(), sizes_.end(), rhs.sizes_.begin()) && data_ == rhs.data_;
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
          return data_.data(index(std::integral_constant < int, 0 >(), t1, indices...));
        }

        /**
         * Get reference to the data point at the (t1, indices...) point
         */
        template<typename ...IndexTypes>
        T &operator()(typename std::enable_if < sizeof...(IndexTypes) == D - 1, size_t >::type t1, IndexTypes ... indices) {
          return data_.data(index(std::integral_constant < int, 0 >(), t1, indices...));
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
        tensor_view < T, D - (sizeof...(IndexTypes)) - 1 > operator()(typename std::enable_if < (sizeof...(IndexTypes) < D - 1), size_t >::type t1, IndexTypes ... indices) {
          std::array < size_t, D - (sizeof...(IndexTypes)) - 1 > sizes;
          size_t s = 1;
          for (int i = 0; i < sizes.size(); ++i) {
            sizes[i] = sizes_[i + sizeof...(IndexTypes) + 1];
            s *= sizes[i];
          }
          return tensor_view < T, D - (sizeof...(IndexTypes)) - 1 >(viewType(data_, s, (index(std::integral_constant < int, 0 >(), t1, indices...))), sizes);
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
          return x;
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
        typename std::enable_if < !std::is_same < S, tensorType >::value && (is_complex < S >::value && !is_complex < T >::value), tensor < S, M > >::type operator*(S scalar) {
          tensor < S, M > x(data_storage < S >(data_.size()), this->sizes_);
          Eigen::Map < Eigen::Matrix < T, 1, Eigen::Dynamic, Eigen::RowMajor > > M1(&data_.data(0), data_.size());
          Eigen::Map < Eigen::Matrix < S, 1, Eigen::Dynamic, Eigen::RowMajor > > M2(&x.data().data(0), data_.size());
          M2 = M1;
          x *= scalar;
          return x;
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
          Eigen::Map < Eigen::Matrix < T, 1, Eigen::Dynamic > > M(&data_.data(0), data_.size());
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
          return x;
        };

        /**
         * Real value Tensor division by complex scalar. Method will create complex Tensor that equals to current real tensor divided by complex scalar.
         */
        template<typename S, int M = D>
        typename std::enable_if < !std::is_same < S, tensorType >::value && (is_complex < S >::value && !is_complex < T >::value), tensor < S, M > >::type operator/(S scalar) {
          tensor < S, M > x(data_storage < S >(data_.size()), this->sizes_);
          Eigen::Map < Eigen::Matrix < T, 1, Eigen::Dynamic, Eigen::RowMajor > > M1(&data_.data(0), data_.size());
          Eigen::Map < Eigen::Matrix < S, 1, Eigen::Dynamic, Eigen::RowMajor > > M2(&x.data().data(0), data_.size());
          M2 = M1;
          x /= scalar;
          return x;
        };

        /**
         * Inplace division
         */
        template<typename S>
        typename std::enable_if < !std::is_same < S, tensorType >::value, tType & >::type operator/=(S scalar) {
          Eigen::Map < Eigen::Matrix < T, 1, Eigen::Dynamic > > M(&data_.data(0), data_.size());
          M /= T(scalar);
          return *this;
        };

        /**
         * @return raw data buffer size
         */
        size_t size() const {
          return data_.size();
        }

        /**
         * For 2D square Tensor compute inverse Tensor.
         * @return inversed Tensor
         */
        template<int M = D>
        typename std::enable_if < M == 2, tensorType >::type
        inverse() {
          if (sizes_[0] != sizes_[1]) {
            throw std::invalid_argument("Can not do inversion of the non-square matrix.");
          }
          tensorType x(*this);
          Eigen::Map < Eigen::Matrix < T, Eigen::Dynamic, Eigen::Dynamic > > Mt(&(x.data().data(0)), sizes_[0], sizes_[1]);
          Mt = Mt.inverse();
          return x;
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
        operator+(const tensor_base < S, D, Ct > &y) {
          tensorType x(*this);
          x += y;
          return x;
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
          !std::is_same < S, T >::value && (std::is_same < S, std::complex < double>>::value || std::is_same < S, std::complex < float>>::value), tensor < S, D > >::type
        operator+(const tensor_base < S, D, Ct > &y) {
          tensor < S, D > x(y);
          x += *this;
          return x;
        };

        /**
         * Compute difference of two tensors of a same type
         */
        template<typename S>
        typename std::enable_if < std::is_same < S, tensorType >::value || std::is_same < S, tensorViewType >::value, tensorType >::type operator-(const S &y) {
          tensorType x(*this);
          x -= y;
          return x;
        };

        /**
         * Compute difference of two tensors of a different type
         */
        template<typename S, typename Ct>
        typename std::enable_if <
          !std::is_same < S, T >::value && (std::is_same < S, std::complex < double>>::value || std::is_same < S, std::complex < float>>::value), tensor < S, D > >::type
        operator-(const tensor_base < S, D, Ct > &y) {
          tensor < S, D > x(y);
          x -= *this;
          return x;
        };

        /**
         * Inplace addition
         */
        template<typename S, typename Ct>
        typename std::enable_if <
          std::is_same < S, T >::value || std::is_same < T, std::complex < double>>::value || std::is_same < T, std::complex < float>>::value, tType & >::type
        operator+=(const tensor_base < S, D, Ct > &y) {
          MatrixMap < T, 1, Eigen::Dynamic > M1(&data_.data(0), data_.size());
          Eigen::Map < const Eigen::Matrix < S, 1, Eigen::Dynamic, Eigen::RowMajor > > M2(&y.data().data(0), y.data().size());
          M1.noalias() += M2;
          return (*this);
        };

        /**
         * Inplace subtraction
         */
        template<typename S>
        typename std::enable_if < std::is_same < S, tensorType >::value || std::is_same < S, tensorViewType >::value, tType & >::type operator-=(const S &y) {
          MatrixMap < T, 1, Eigen::Dynamic > M1(&data_.data(0), data_.size());
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
          if (sizes_[0] != y.sizes_[1] && sizes_[1] != y.sizes_[0]) {
            throw std::invalid_argument("Can not do multiplication. Dimensions missmatches.");
          }
          tensorType x(*this);
          MatrixMap < T > M1(&x.data().data(0), sizes_[0], sizes_[1]);
          Eigen::Map < const Eigen::Matrix < T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > > M2(&y.data().data(0), sizes_[0], sizes_[1]);
          M1 *= M2;
          return x;
        };

        /**
         * @return Eigen matrix representation for 2D Tensor
         */
        template<int M = D>
        typename std::enable_if < M == 2, MatrixMap < T > >::type
        matrix() {
          return MatrixMap < T >(&data().data(0), sizes_[0], sizes_[1]);
        };



        /// sizes for each dimension
        const std::array < size_t, D > &sizes() const { return sizes_; };

        /// offset multipliers for each dimension
        const std::array < size_t, D > &acc_sizes() const { return acc_sizes_; };

        /// dimension of the tensor
        int dimension() const { return D; }

        /// const reference to the data storage
        const C &data() const { return data_; }

        /// reference to the data storage
        C &data() { return data_; }

        // these methods should be private but they are public for the test purposes
      public:
        /// return index in the raw buffer for specified indices
        template<int M, typename Ti, typename ...Ts>
        size_t index(std::integral_constant < int, M > i, const Ti &t1, const Ts &...ts) const {
          return t1 * acc_sizes_[i.value] + index(std::integral_constant < int, M + 1 >(), std::forward < const Ts & >(ts)...);
        }

        /// return index in the raw buffer for the last index
        template<int M, typename Ti>
        size_t index(std::integral_constant < int, M > i, const Ti &t1) const {
          return t1 * acc_sizes_[i.value];
        }

      private:
        /**
         * compute offset multiplier for each dimension
         */
        void fill_acc_sizes() {
          int k = 1;
          for (size_t &i : acc_sizes_) {
            i = 1;
            for (int j = k; j < sizes_.size(); ++j) {
              i *= sizes_[j];
            }
            ++k;
          }
        }
        /// compte size for the specific dimensions
        size_t size(const std::array < size_t, D > &sizes) {
          size_t res = 1;
          for (int i = 0; i < D; ++i) {
            res *= sizes[i];
          }
          return res;
        }
      };

    }
  }
}

#endif //GF2_TENSOR_H
