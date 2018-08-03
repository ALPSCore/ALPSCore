/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_GF_TENSORBASE_H
#define ALPSCORE_GF_TENSORBASE_H

#include <alps/numeric/tensors/mpi/mpi_shared_allocator.hpp>

namespace alps {
  namespace numerics {
    namespace detail {

      /**
       * Simple data allocator
       * @tparam T - datatype
       */
      template <typename T>
      class simple_allocator {

      public:
        /**
         * Allocate new array of requested size and return pointer to the first element
         *
         * @param size - size to allocate
         * @return pointer to the first elemnt of allocated data array
         */
        T* allocate(size_t size) {
          return new T[size];
        }

        /**
         * release memory
         * @param data
         */
        void deallocate(T* data) {
          delete [] data;
        }

        void lock(){}
        void release(){}
        bool locked() {return true;}
      };

      // Forward declarations
      template<typename T>
      class data_view;

      /**
       * @brief TensorBase class implements basic multi-dimensional operations.
       * All arithmetic operations are performed by Eigen library
       */
      template<typename T, typename Allocator = simple_allocator<T> >
      class data_storage {
      private:
        /// internal data storage
        T* data_;
        /// size of buffer
        size_t size_;
        /// dataspace allocator
        Allocator allocator_;
        /// Check the data is locked in case of shared memory access
        void check_lock() {
          if(!allocator_.locked()) {
            std::cerr<<"You are trying to access non-locked tensor with non-const method."<<std::endl;
          }
        }
      public:

        /// create data_dtorage from other storage object by copying data into vector
        template<typename T2, typename A2>
        data_storage(const data_storage<T2, A2> & storage, const Allocator & allocator = Allocator()) : size_(storage.size()), allocator_(allocator) {
          data_ = allocator_.allocate(storage.size());
          std::copy(storage.data(), storage.data() + storage.size(), data_);
        };
        template<typename T2, typename A2>
        data_storage(data_storage<T2, A2> && storage, const Allocator & allocator = Allocator()) : size_(storage.size()), allocator_(allocator) {
          data_ = allocator_.allocate(storage.size());
          std::copy(storage.data(), storage.data() + storage.size(), data_);
        };

        /// Copy constructor
        data_storage(const data_storage<T, Allocator>& rhs) : size_(rhs.size_), allocator_(rhs.allocator_) {
          data_ = allocator_.allocate(rhs.size());
          std::copy(rhs.data(), rhs.data() + rhs.size(), data_);
        };
        /// Move Constructor
        data_storage(data_storage<T, Allocator>&& rhs) : size_(rhs.size_), allocator_(rhs.allocator_) {
          data_ = allocator_.allocate(rhs.size());
          std::copy(rhs.data(), rhs.data() + rhs.size(), data_);
        };
        /// Copy assignment
        data_storage<T, Allocator>& operator=(const data_storage<T, Allocator>& rhs) {
          if(size() != rhs.size()) {
            resize(rhs.size());
          }
          std::copy(rhs.data(), rhs.data() + rhs.size(), data_);
          return *this;
        };
        /// Move assignment
        data_storage<T, Allocator>& operator=(data_storage<T, Allocator>&& rhs) {
          if(size() != rhs.size()) {
            resize(rhs.size());
          }
          std::copy(rhs.data(), rhs.data() + rhs.size(), data_);
          return *this;
        };

        template<typename T2, typename A2>
        data_storage<T, Allocator>& operator=(const data_storage<T2, A2>& rhs) {
          static_assert(std::is_convertible<T2, T>::value, "Can't perform assignment: T2 can be casted into T");
          if(size() != rhs.size()) {
            resize(rhs.size());
          }
          std::copy(rhs.data(), rhs.data() + rhs.size(), data_);
          return *this;
        };
        /// Create data_dtorage from the view object by copying data into vector
        template<typename T2>
        data_storage(const data_view<T2> & view, const Allocator & a = Allocator())  : size_(view.size()), allocator_(a) {
          static_assert(std::is_convertible<T2, T>::value, "View type can not be converted into storage");
          data_ = allocator_.allocate(view.size());
          std::copy(view.data(), view.data() + view.size(), data_);
        }
        /// Move-Create DataStorage from the view object by copying data into vector
        template<typename T2>
        data_storage(data_view<T2> && view) noexcept  : size_(view.size()){
          data_ = allocator_.allocate(view.size());
          std::copy(view.data(), view.data() + view.size(), data_);
        };
        /// Create data storage from raw buffer by data copying
        data_storage(const T *data, size_t size) : size_(size) {
          data_ = allocator_.allocate(size);
          std::copy(data, data + size, data_);
        }
        /// Create empty storage of size %size%
//        explicit data_storage(size_t size) : size_(size) {
//          data_ = allocator_.allocate(size);
//          std::fill(data_, data_+size, T(0));
//        }

        explicit data_storage(size_t size, const Allocator& allocator = Allocator()) : size_(size), allocator_(allocator) {
          data_ = allocator_.allocate(size);
          std::fill(data_, data_+size, T(0));
        }

        virtual ~data_storage() {
          allocator_.deallocate(data_);
        }

        /// @return reference to the data at point i
        inline T& data(size_t i) {
#ifndef NDEBUG
          check_lock();
#endif
          return data_[i];
        };
        /// @return const-reference to the data at point i
        inline const T& data(size_t i) const {return data_[i];};
        /// bracket operators
        inline const T& operator()(size_t i) const {return data_[i];};
        inline T& operator()(size_t i) {
#ifndef NDEBUG
          check_lock();
#endif
          return data_[i];
        };
        /// @return data size
        size_t size() const {return size_;}
        /// @return const-reference to stored vector
        const T* data() const {return data_;}
        /// @return reference to stored vector
        T* data() {
#ifndef NDEBUG
          check_lock();
#endif
          return data_;}
        /// perform data-access lock
        void lock(){allocator_.lock();}
        /// release internal data
        void release(){allocator_.release();}

        /// Data-storage resize
        void resize(size_t new_size) {
          // create copy of the allocator
          // we need it to ensure that in case of MPI shared memory allocator
          // the communication window for old data buffer will be keeped and new window will be created for
          // new data buffer
          Allocator allocator(allocator_);
          // Allocate memory for new data buffer. New communication window will be created
          T* data = allocator.allocate(new_size);
          // Copy data from old data buffer to new. If new size is smaller we will truncate data
          std::copy(data_, data_ + std::min(size_, new_size), data);
          // Deallocate old data buffer. Communication window should be released
          allocator_.deallocate(data_);
          // Update storage fields
          data_ = data;
          size_ = new_size;
          allocator_ = allocator;
        }

        /**
         * Data Storage comparison. Two data storages are equal if they have the same size
         * and all elements are equal to each other
         *
         * @tparam T2 - datatype of rhs storage
         * @param r   - rhs storage
         */
        template<typename T2, typename A2>
        bool operator==(const data_storage<T2, A2> &r) const {
          return r.size() == size() && std::equal(r.data(), r.data() + r.size(), data_);
        }

        /// Comparison against DataView
        template<typename T2>
        bool operator==(const data_view<T2>& r) const {
          return r == *this;
        };
      };
    }
    template<typename T>
    using simple_storage = detail::data_storage<T, detail::simple_allocator<T> >;
#ifdef ALPS_HAVE_MPI
    template<typename T>
    using shared_storage = detail::data_storage<T, detail::mpi_shared_allocator<T> >;
#endif
  }
}

#endif //ALPSCORE_GF_TENSORBASE_H
