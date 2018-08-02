/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_GF_TENSORBASE_H
#define ALPSCORE_GF_TENSORBASE_H

#include <alps/config.hpp>

#ifdef ALPS_HAVE_MPI
#include <mpi.h>
#endif


namespace alps {
  namespace numerics {
    namespace detail {

      template <typename T>
      class SimpleAllocator {

      public:
        T* allocate(size_t size) {
          return new T[size];
        }

        void deallocate(T* data) {
          delete [] data;
        }

        void lock(){}
        void release(){}
      };

#ifdef ALPS_HAVE_MPI
      template <typename T>
      class MPISharedAllocator {
      public:

        MPISharedAllocator(MPI_Comm& comm, MPI_Win& win) : _comm(&comm), _win(&win) {

        };

        T* allocate(size_t size) {
          void *mem;
          // compute local size for array
          size_t nloc = local_size(size);
          // allocate shared memory
          MPI_Win_allocate_shared(nloc * sizeof(T), sizeof(T), MPI_INFO_NULL, *_comm, &mem, _win);
          MPI_Aint alloc_length;
          int disp_unit;
          // get pointer to the 0-th element of the global array
          MPI_Win_shared_query(*_win, 0, &alloc_length, &disp_unit, &mem);
          return (T*) mem;
        }

        void deallocate(T* data) {
          // we do not release anything.
          // one should explicitly release MPI-window in the very original place.
        }
        void lock(){
          MPI_Win_lock_all(MPI_MODE_NOCHECK , *_win);
        }
        void release(){
          MPI_Win_sync(*_win);
          MPI_Win_unlock_all(*_win);
        }
      private:
        /**
         * compute local size for global array
         * @param n -- global size
         * @return local size
         */
        size_t local_size(size_t n) const {
          int s, r;
          MPI_Comm_size(*_comm, &s);
          MPI_Comm_rank(*_comm, &r);
          size_t nloc = n / s;
          if(n%s > r) nloc++;
          return nloc;
        }

        MPI_Comm *_comm;
        MPI_Win *_win;
      };
#endif

      // Forward declarations
      template<typename T>
      class data_view;

      /**
       * @brief TensorBase class implements basic multi-dimensional operations.
       * All arithmetic operations are performed by Eigen library
       */
      template<typename T, typename Allocator = SimpleAllocator<T> >
      class data_storage {
      private:
        /// internal data storage
        T* data_;
        /// size of buffer
        size_t size_;
        /// dataspace allocator
        Allocator allocator_;
        /// Lock data in case of shared memory access
        bool lock_;
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
          assert(size() == rhs.size());
          data_ = allocator_.allocate(rhs.size());
          std::copy(rhs.data(), rhs.data() + rhs.size(), data_);
          return *this;
        };
        /// Move assignment
        data_storage<T, Allocator>& operator=(data_storage<T, Allocator>&& rhs) {
          assert(size() == rhs.size());
          data_ = allocator_.allocate(rhs.size());
          std::copy(rhs.data(), rhs.data() + rhs.size(), data_);
          return *this;
        };

        template<typename T2, typename A2>
        data_storage<T, Allocator>& operator=(const data_storage<T2, A2>& rhs) {
          static_assert(std::is_convertible<T2, T>::value, "Can't perform assignment: T2 can be casted into T");
          assert(size() == rhs.size());
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
          assert(lock_);
          return data_[i];};
        /// @return const-reference to the data at point i
        inline const T& data(size_t i) const {return data_[i];};
        /// bracket operators
        inline const T& operator()(size_t i) const {return data_[i];};
        inline T& operator()(size_t i) {
          assert(lock_);
          return data_[i];
        };
        /// @return data size
        size_t size() const {return size_;}
        /// @return const-reference to stored vector
        const T* data() const {return data_;}
        /// @return reference to stored vector
        T* data() {
          assert(lock_);
          return data_;}
        /// perform data-access lock
        void lock(){lock_ = true; allocator_.lock();}
        /// release internal data
        void release(){lock_ = false; allocator_.release();}

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
    using simple_storage = detail::data_storage<T, detail::SimpleAllocator<T> >;
#ifdef ALPS_HAVE_MPI
    template<typename T>
    using shared_storage = detail::data_storage<T, detail::MPISharedAllocator<T> >;
#endif
  }
}

#endif //ALPSCORE_GF_TENSORBASE_H
