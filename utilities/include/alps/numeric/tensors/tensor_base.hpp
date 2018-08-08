/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_GF_TENSOR_H
#define ALPSCORE_GF_TENSOR_H


#include <array>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <alps/type_traits/index_sequence.hpp>
#include <alps/type_traits/are_all_integrals.hpp>
#include <alps/numeric/tensors/data_view.hpp>


namespace alps {
  namespace numerics {
    namespace detail {
      /**
       *
       * @tparam T
       * @tparam St
       */
      template<typename T, typename St>
      struct is_storage {
        static constexpr bool value = std::is_same < St, simple_storage < T > > ::value ||
            std::is_same < St, data_view < T > > ::value;
      };

      /**
       * Check that all values in pack are true
       *
       * @tparam IndicesTypes - types of indices template parameter pack
       */
      template <bool...> struct bool_pack;
      template <bool... v>
      using all_true = std::is_same<bool_pack<true, v...>, bool_pack<v..., true>>;

      /**
       * Base Tensor Class
       *
       * @tparam T - datatype, should be scalar
       * @tparam D - dimension of tensor
       * @tparam C - type of the container, either DataStorage or DataView
       */
      template<typename T, size_t D, typename C>
      class tensor_base;
    }
      /**
       * Definition of Tensor with storage
       */
      template<typename T, size_t D>
      using tensor = detail::tensor_base < T, D, detail::data_storage < T > >;
#ifdef ALPS_HAVE_SHARED_ALLOCATOR
      /**
       * Definition of Tensor with mpi3 shared storage
       */
      template<typename T, size_t D>
      using shared_tensor = detail::tensor_base < T, D, shared_storage< T > >;
#endif
      /**
       * Definition of Tensor as view of existent data array
       */
      template<typename T, size_t D>
      using tensor_view = detail::tensor_base < T, D, detail::data_view < T > >;

    namespace detail {
      template<typename X, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
      using MatrixMap =  Eigen::Map < Eigen::Matrix < X, Rows, Cols, Eigen::RowMajor > >;
      template<typename X, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
      using ConstMatrixMap =  Eigen::Map <const Eigen::Matrix < X, Rows, Cols, Eigen::RowMajor > >;

      /**
       * @brief Tensor class for raw data storage and performing the basic arithmetic operations
       *
       * @author iskakoff
       */
      template<typename T, size_t Dim, typename Container>
      class tensor_base {
      public:
        // types definitions
        typedef T prec;
      private:
        typedef data_view < T > viewType;
        typedef data_view < const typename std::remove_const<T>::type > constViewType;
        typedef data_view < typename std::remove_const<T>::type > nonconstViewType;
        typedef data_storage < T > storageType;
        typedef data_storage < const typename std::remove_const<T>::type > constStorageType;
        typedef data_storage < typename std::remove_const<T>::type > nonconstStorageType;
        /// current Tensor type
        typedef tensor_base < T, Dim, Container > tType;
        /// Tensor type with storage
        typedef tensor < T, Dim > tensorType;
        /// view Tensor type
        typedef tensor_view < T, Dim > tensorViewType;
        /// generic tensor type
        template<typename St>
        using   genericTensor = tensor_base < T, Dim, St >;

      private:
        // fields definitions
        /// tensor dimension
        static size_t constexpr dim = Dim;
        /// data storage object
        Container storage_;
        /// stored sizes for each dimensions
        std::array < size_t, Dim > shape_;
        /// offset multiplier for each dimension
        std::array < size_t, Dim > acc_sizes_;

      public:

        /**
         * Create empty tensor.
         */
        tensor_base() : storage_(0) {};

        /**
         * @param container - internal storage container
         * @param sizes
         */
        tensor_base(Container &&container, const std::array < size_t, Dim >& sizes) : storage_(container), shape_(sizes) {
          static_assert(is_storage< T, Container>::value, "Should be either data_storage or data_view type");
          assert(storage_.size() == size());
          fill_acc_sizes();
        }

        tensor_base(Container &container, const std::array < size_t, Dim >& sizes) : storage_(container), shape_(sizes) {
          static_assert(is_storage< T, Container>::value, "Should be either data_storage or data_view type");
          assert(storage_.size() == size());
          fill_acc_sizes();
        }

        template<typename Container2>
        tensor_base(Container2 &container, size_t size, size_t offset, const std::array < size_t, Dim >& sizes) :
            storage_(container, size, offset), shape_(sizes) {
          assert(storage_.size() == this->size());
          fill_acc_sizes();
        }

        template<typename Container2>
        tensor_base(const Container2 &container, size_t size, size_t offset, const std::array < size_t, Dim >& sizes) :
            storage_(container, size, offset), shape_(sizes) {
          assert(storage_.size() == this->size());
          fill_acc_sizes();
        }

        template<typename...Indices>
        tensor_base(typename std::enable_if< all_true<std::is_convertible<Indices, std::size_t>::value...>::value, Container >::type &container,
                    size_t size1, Indices...sizes) : tensor_base(container, {{size1, size_t(sizes)...}}) {}

        /**
         * Create tensor from the existent data. All operation will be performed on the data stored in <data> parameter
         * if DataView storage is used. In case of DataStorage storage all data will be copied into raw vector.
         *
         * @param data  - pointer to the raw data buffer
         * @param sizes - array with sizes for each dimension
         */
        tensor_base(T *data, const std::array < size_t, Dim > & sizes) : storage_(data, size(sizes)), shape_(sizes) {
          fill_acc_sizes();
        }

        template<typename...Indices>
        tensor_base(T *data, size_t size1, Indices...sizes) : tensor_base(data, {{size1, size_t(sizes)...}}) {}

        /**
         * Create empty Tensor with provided sizes for each dimensions
         *
         * @tparam X type of storage to be created. Should always be DataStorage
         * @param sizes - array of data dimensions
         */
        template<typename X = Container>
        tensor_base(typename std::enable_if < std::is_same < X, data_storage < T > >::value,
                        const std::array < size_t, Dim > & >::type sizes) : storage_(size(sizes)), shape_(sizes) {
          fill_acc_sizes();
        }

        template<typename X = Container, typename...Indices>
        tensor_base(typename std::enable_if < std::is_same < X, data_storage < T > >::value,
          size_t>::type size1, Indices...sizes) : storage_(size({{size1, size_t(sizes)...}})), shape_({{size1, size_t(sizes)...}}) {
          static_assert(sizeof...(Indices) + 1 == Dim, "Wrong dimension");
          fill_acc_sizes();
        }

        // this constructor create a view of other tensor. that is why rhs is not const
        template<typename St = Container>
        tensor_base(typename std::enable_if<std::is_same<St, viewType>::value, tensorType>::type& rhs) :
          storage_(rhs.storage()), shape_(rhs.shape()), acc_sizes_(rhs.acc_sizes()) {}

        /// copy constructor
        tensor_base(const tType& rhs) = default;
        /// move constructor
        tensor_base(tType &&rhs) = default;
        /// copy constructor
        template<typename T2, typename St, typename = std::enable_if<std::is_same<Container, storageType>::value, void >>
        tensor_base(const tensor_base<T2, Dim, St> &rhs) : storage_(rhs.storage()), shape_(rhs.shape()), acc_sizes_(rhs.acc_sizes()) {}
        /// move constructor
        template<typename T2, typename St, typename = std::enable_if<std::is_same<Container, storageType>::value, void >>
        tensor_base(tensor_base<T2, Dim, St> &&rhs) noexcept: storage_(rhs.storage()), shape_(rhs.shape()), acc_sizes_(rhs.acc_sizes()) {}

        /// Different type assignment
        template<typename T2, typename St>
        tensor_base < T, Dim, Container > &operator=(const tensor_base < T2, Dim, St> &rhs){
          assert(size()==rhs.size());
          storage_ = rhs.storage();
          return *this;
        };
        /// Copy assignment
        tensor_base < T, Dim, Container > &operator=(const tensor_base < T, Dim, Container > &rhs) = default;
        /// Move assignment
        tensor_base < T, Dim, Container > &operator=(tensor_base < T, Dim, Container > &&rhs) = default;
        /// compare tensors
        template<typename T2, typename St>
        bool operator==(const tensor_base<T2, Dim, St>& rhs) const {
          return std::equal(shape_.begin(), shape_.end(), rhs.shape().begin()) && storage_ == rhs.storage();
        }

        /**
         * Get data point for the specific set of indices
         * @tparam IndexTypes - types of tail indices
         * @param t1          - head index
         * @param indices     - tail indices
         * @return value of tensor at the (t1, indices...) point
         */
        template<typename ...IndexTypes>
        const T & operator()(typename std::enable_if < sizeof...(IndexTypes) == Dim - 1, size_t >::type t1, IndexTypes ... indices) const {
          return storage_.data(index(t1, indices...));
        }

        /**
         * Get reference to the data point at the (t1, indices...) point
         */
        template<typename ...IndexTypes>
        T &operator()(typename std::enable_if < sizeof...(IndexTypes) == Dim - 1, size_t >::type t1, IndexTypes ... indices) {
          return storage_.data(index(t1, indices...));
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
        tensor_view < T, Dim - (sizeof...(IndexTypes)) - 1 > operator()(typename std::enable_if < (sizeof...(IndexTypes) < Dim - 1),
            size_t >::type t1, IndexTypes ... indices) {
          std::array < size_t, Dim - (sizeof...(IndexTypes)) - 1 > sizes;
          size_t s = new_size(sizes);
          return tensor_view < T, Dim - (sizeof...(IndexTypes)) - 1 > ( storage_, s, index(t1, indices...), sizes);
        }

        template<typename ...IndexTypes>
        tensor_view <const typename std::remove_const<T>::type, Dim - (sizeof...(IndexTypes)) - 1 > operator()
            (typename std::enable_if < (sizeof...(IndexTypes) < Dim - 1), size_t >::type t1, IndexTypes ... indices) const {
          std::array < size_t, Dim - (sizeof...(IndexTypes)) - 1 > sizes;
          size_t s = new_size(sizes);
          return tensor_view <const typename std::remove_const<T>::type, Dim - (sizeof...(IndexTypes)) - 1 >
              (storage_, s, index(t1, indices...), sizes );
        }

        /*
         * Basic arithmetic operations
         */
        /**
         * Multiplication by scalar. This method checks the resulting type of Tensor based on the scalar type and tensor type.
         *
         * @tparam S     - type of scalar multiplier
         * @param scalar - scalar factor
         * @return New tensor equal to the current tensor multiplied by scalar
         */
        template<typename S>
        typename std::enable_if < !std::is_same < S, tensorType >::value, tensor_base< decltype(S{} + T{}), Dim,
            data_storage< decltype(S{} * T{}) > > >::type operator*(S scalar) const {
          tensor_base< decltype(S{} + T{}), Dim, data_storage< decltype(S{} * T{}) > > x(*this);
          return (x *= static_cast<decltype(S{} + T{})>(scalar));
        };

        /**
         * Two tensors multiplication. We need to decide whether it should be element-wise or more sofisticated tensor contraction.
         *
         * @tparam S     - type of the right hand side tensor
         * @param rhs - right hand side tensor
         * @return result of two tensor multiplication
         */
        template<typename S>
        typename std::enable_if < std::is_same < S, tensorType >::value, tensorType >::type operator*(const S& rhs) const {
          tensorType x(*this);
          return x*=rhs;
        };

        /**
         * Inplace tensor scaling
         */
        template<typename S>
        typename std::enable_if < !std::is_same < S, tensorType >::value, tType & >::type operator*=(S scalar) {
          static_assert(std::is_convertible<S, T>::value, "Can't perform inplace multiplication: S can be casted into T");
          Eigen::Map < Eigen::Matrix < T, 1, Eigen::Dynamic > > M(&storage_.data(0), storage_.size());
          M *= T(scalar);
          return *this;
        };

        /**
         * Inplace tensor multiplication
         */
        template<typename S>
        typename std::enable_if < std::is_same < S, tensorType >::value, tensorType & >::type operator*=(const S& rhs) {
          Eigen::Map < Eigen::Array < T, 1, Eigen::Dynamic > > M1(&storage_.data(0), storage_.size());
          Eigen::Map < const Eigen::Array < T, 1, Eigen::Dynamic > > M2(&rhs.storage().data(0), rhs.storage().size());
          M1*=M2;
          return *this;
        };

        /**
         * Tensor inversed scaling
         */
        template<typename S>
        typename std::enable_if < !std::is_same < S, tensorType >::value, tensor_base< decltype(S{} + T{}), Dim,
            data_storage< decltype(S{} * T{}) > > >::type operator/(S scalar) const {
          tensor_base< decltype(S{} + T{}), Dim, data_storage< decltype(S{} * T{}) > >  x(*this);
          return (x /= scalar);
        };

        /**
         * Inplace division
         */
        template<typename S>
        typename std::enable_if < !std::is_same < S, tensorType >::value, tType & >::type operator/=(S scalar) {
          static_assert(std::is_convertible<S, T>::value, "Can not perform inplace division: S can be casted into T");
          Eigen::Map < Eigen::Matrix < T, 1, Eigen::Dynamic > > M(&storage_.data(0), storage_.size());
          M *= T(1.0)/T(scalar);
          return *this;
        };

        /**
         * Negate tensor
         */
        tensor_base < T, Dim, Container > operator-() const {
          return *this * T(-1.0);
        };

        /**
         * Set data to 0
         */
        void set_zero() {
          set_number(T(0));
        }

        /**
         * Assign all the values in the tensor to the specific scalar
         *
         * @tparam num_type - scalar value type
         * @param value     - scalar value for all elements in the tensor
         */
        template<typename num_type>
        void set_number(num_type value) {
          static_assert(std::is_convertible<num_type, T>::value, "Can not assign value to the tensor. Value can not be cast into the tensor value type.");
          std::fill_n(data(), size(), T(value));
        }

        /**
         * @return raw data buffer size
         */
        size_t size() const {
          return storage_.size();
        }

        /**
         * @return the total number of elements in the tensor
         */
        size_t num_elements() const {
          return storage_.size();
        }

        /**
         * For 2D square Tensor compute inverse Tensor.
         * @return inversed Tensor
         */
        template<size_t M = Dim>
        typename std::enable_if < M == 2, tensorType >::type
        inverse() {
          if (shape_[0] != shape_[1]) {
            throw std::invalid_argument("Can not do inversion of the non-square matrix.");
          }
          tensorType x(*this);
          Eigen::Map < Eigen::Matrix < T, Eigen::Dynamic, Eigen::Dynamic > > Mt(&(x.storage().data(0)), shape_[0], shape_[1]);
          Mt = Mt.inverse();
          return x;
        };

        /**
         * Compute sum of two tensors
         *
         * @tparam S  - data type of rhs tensor
         * @tparam Ct - storage type of rhs tensor
         * @param  rhs  - rhs tensor
         * @return new tensor object that equals to sum of current tensor and rhs tensor
         */
        template<typename S, typename Ct>
        tensor < decltype(S{} + T{}), Dim > operator+(const tensor_base < S, Dim, Ct > &rhs) {
          tensor < decltype(S{} + T{}), Dim > x(*this);
          return (x += rhs);
        };

        /**
         * Compute difference of two tensors of a same type
         */
        template<typename S, typename Ct>
        tensor < decltype(S{} - T{}), Dim > operator-(const tensor_base < S, Dim, Ct > &rhs) {
          tensor < decltype(S{} - T{}), Dim > x(*this);
          return (x -= rhs);
        };

        /**
         * Inplace addition
         */
        template<typename S, typename Ct>
        typename std::enable_if <
          std::is_same < S, T >::value || std::is_same < T, std::complex < double>>::value
          || std::is_same < T, std::complex < float>>::value, tType & >::type operator+=(const tensor_base < S, Dim, Ct > &y) {
          MatrixMap < T, 1, Eigen::Dynamic > M1(&storage_.data(0), storage_.size());
          ConstMatrixMap < S, 1, Eigen::Dynamic > M2(&y.storage().data(0), y.storage().size());
          M1.noalias() += M2;
          return (*this);
        };

        /**
         * Inplace subtraction
         */
        template<typename S>
        typename std::enable_if < std::is_same < S, tensorType >::value ||
            std::is_same < S, tensorViewType >::value, tType & >::type operator-=(const S &y) {
          MatrixMap < T, 1, Eigen::Dynamic > M1(&storage_.data(0), storage_.size());
          ConstMatrixMap < T, 1, Eigen::Dynamic> M2(&y.storage().data(0), y.storage().size());
          M1.noalias() -= M2;
          return (*this);
        };

        /**
         * Compute a dot product of two 2D tensors
         */
        tensorType dot(const tensorType &y) const {
          static_assert(Dim == 2, "Can not do inversion for not 2D tensor.");
          if (shape_[1] != y.shape_[0]) {
            throw std::invalid_argument("Can not do multiplication. Dimensions missmatches.");
          }
          tensorType x({{shape_[0], y.shape_[1]}});
          ConstMatrixMap < T, Eigen::Dynamic, Eigen::Dynamic > M1(&storage().data(0), shape_[0], shape_[1]);
          ConstMatrixMap < T, Eigen::Dynamic, Eigen::Dynamic > M2(&y.storage().data(0), y.shape()[0], y.shape()[1]);
          MatrixMap < T > M3(&x.storage().data(0), x.shape()[0], x.shape()[1]);
          M3 = M1*M2;
          return x;
        };

        /**
         * @return Eigen matrix representation for 2D Tensor
         */
        MatrixMap < T > matrix() {
          static_assert(Dim == 2, "Can not return Eigen matrix view for not 2D tensor.");
          return MatrixMap < T >(&storage().data(0), shape_[0], shape_[1]);
        };

        /// sizes for each dimension
        const std::array < size_t, Dim > &shape() const { return shape_; };

        /// reshape with index list
        template<typename ...Inds>
        typename std::enable_if<are_all_integrals<Inds...>::value>::type reshape(Inds...inds) {
          static_assert(sizeof...(Inds) == Dim, "New shape should have the same dimension.");
          std::array<size_t, Dim> shape = {{size_t(inds)...}};
          reshape(shape);
        }

        /// reshape tensor object
        template<typename X = Container>
        typename std::enable_if<std::is_same < X, data_storage < T > >::value, void>::type reshape(const std::array<size_t, Dim>& shape) {
          size_t new_size = size(shape);
          storage_.resize(new_size);
          shape_ = shape;
          fill_acc_sizes();
        }

        /// reshape tensor view object
        template<typename X = Container>
        typename std::enable_if<std::is_same < X, data_view < T > >::value, void>::type reshape(const std::array<size_t, Dim>& shape) {
          size_t new_size = size(shape);
          if(new_size != size()) {
            throw std::invalid_argument("Wrong size. Can't reshape tensor.");
          }
          shape_ = shape;
          fill_acc_sizes();
        }

        /// offset multipliers for each dimension
        const std::array < size_t, Dim > &acc_sizes() const { return acc_sizes_; };

        /// dimension of the tensor
        static size_t dimension() { return Dim; }

        /// const pointer to the internal data
        const prec *data() const { return storage_.data(); }

        /// pointer to the internal data
        prec *data() { return storage_.data(); }

        /// const reference to the data storage
        const Container &storage() const { return storage_; }

        /// reference to the data storage
        Container &storage() { return storage_; }

        /// return index in the raw buffer for specified indices
        template<typename ...Indices>
        inline size_t index(const Indices&...indices) const {
          return index_impl(make_index_sequence<sizeof...(Indices)>(), size_t(indices)...);
        }

      private:
        /**
         * Internal implementation of indexing
         */
        template<size_t... I, typename ...Indices>
        inline size_t index_impl(index_sequence<I...>, const Indices&... indices) const {
          std::array<size_t, sizeof...(Indices)> a{{indices * acc_sizes_[I] ...}};
          return std::accumulate(a.begin(), a.end(), size_t(0), std::plus<size_t>());
        }
        /**
         * compute offset multiplier for each dimension
         */
        void fill_acc_sizes() {
          acc_sizes_[shape_.size()-1] = 1;
          for(int k = int(shape_.size()) - 2; k >= 0; --k)
            acc_sizes_[k] = acc_sizes_[k+1] * shape_[k+1];
        }
        /// compte size for the specific dimensions
        size_t size(const std::array < size_t, Dim > &shape) {
          return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
        }
        /// compute sizes for sliced tensor
        template<size_t M>
        inline size_t new_size(std::array < size_t, M >& sizes) const {
          size_t s = 1;
          for (size_t i = 0; i < sizes.size(); ++i) {
            sizes[i] = shape_[i + Dim - M];
            s *= sizes[i];
          }
          return s;
        }
      };

    }
  }
}

#endif //ALPSCORE_GF_TENSOR_H
