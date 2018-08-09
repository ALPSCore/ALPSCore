/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_GF_TENSORBASE_H
#define ALPSCORE_GF_TENSORBASE_H

#include <vector>

namespace alps {
  namespace numerics {
    namespace detail {

      // Forward declarations
      template<typename T>
      class data_view;

      /**
       * @brief Internal data storage class for tensors
       *
       * @tparam T  the scalar type
       * @tparam Cont  abstraction of the data storage (default vector)
       */
      template<typename T, typename Cont = std::vector<typename std::remove_const<T>::type> >
      class data_storage {
      private:
        /// internal data storage
        Cont data_;
      public:

        /// create data_dtorage from other storage object by copying data into vector
        template<typename T2, typename C2>
        data_storage(const data_storage<T2, C2> & storage) : data_(storage.size()) {
          std::copy(storage.data(), storage.data() + storage.size(), data());
        };
        template<typename T2, typename C2>
        data_storage(data_storage<T2, C2> && storage) : data_(storage.size()) {
          std::copy(storage.data(), storage.data() + storage.size(), data());
        };

        /// Copy constructor
        data_storage(const data_storage<T, Cont>& rhs) : data_(rhs.data_) {};
        /// Move Constructor
        data_storage(data_storage<T, Cont>&& rhs) : data_(rhs.data_) {};
        /// Copy assignment
        data_storage<T, Cont>& operator=(const data_storage<T, Cont>& rhs) {
          if(size() != rhs.size()) {
            resize(rhs.size());
          }
          std::copy(rhs.data(), rhs.data() + rhs.size(), data());
          return *this;
        };
        /// Move assignment
        data_storage<T, Cont>& operator=(data_storage<T, Cont>&& rhs) {
          if(size() != rhs.size()) {
            resize(rhs.size());
          }
          std::copy(rhs.data(), rhs.data() + rhs.size(), data());
          return *this;
        };

        /**
         * General tensor assignment 
         *
         * @tparam T2  - rhs value type
         * @tparam C2  - rhs storage container type
         */
        template<typename T2, typename C2>
        data_storage<T, Cont>& operator=(const data_storage<T2, C2>& rhs) {
          static_assert(std::is_convertible<T2, T>::value, "Can not perform assignment: T2 can not be cast into T");
          if(size() != rhs.size()) {
            resize(rhs.size());
          }
          std::copy(rhs.data(), rhs.data() + rhs.size(), data());
          return *this;
        };
        /// Create data_dtorage from the view object by copying data into underlying container
        template<typename T2>
        data_storage(const data_view<T2> & view)  : data_(view.size()) {
          static_assert(std::is_convertible<T2, T>::value, "View type can not be converted into storage");
          std::copy(view.data(), view.data() + view.size(), data());
        }
        /// Move-Create DataStorage from the view object by copying data into underlying container
        template<typename T2>
        data_storage(data_view<T2> && view) noexcept  : data_(view.size()){
          std::copy(view.data(), view.data() + view.size(), data());
        };
        /// Create data storage from raw buffer by data copying
        data_storage(const T *data, size_t size) : data_(size) {
          std::copy(data, data + size, this->data());
        }
        /// Create empty storage of size %size%
        explicit data_storage(size_t size) : data_(size) {
          std::fill(data(), data()+size, T(0));
        }

        /// @return reference to the data at point i
        inline T& data(size_t i) {
          return data_[i];
        };
        /// @return const-reference to the data at point i
        inline const T& data(size_t i) const {return data_[i];};
        /// bracket operators
        inline const T& operator()(size_t i) const {return data_[i];};
        inline T& operator()(size_t i) {
          return data_[i];
        };
        /// @return data size
        size_t size() const {return data_.size();}
        /// @return const-reference to stored vector
        const T* data() const {return data_.data();}
        /// @return reference to stored vector
        T* data() {return data_.data();}
        /// Data-storage resize
        void resize(size_t new_size) {
          data_.resize(new_size);
        }

        /**
         * Data Storage comparison. Two data storages are equal if they have the same size
         * and all elements are equal to each other
         *
         * @tparam T2 - datatype of rhs storage
         * @tparam C2 - data storage type of rhs storage
         * @param r   - rhs storage
         */
        template<typename T2, typename C2>
        bool operator==(const data_storage<T2, C2> &r) const {
          return r.size() == size() && std::equal(r.data(), r.data() + r.size(), data());
        }

        /// Comparison against DataView
        template<typename T2>
        bool operator==(const data_view<T2>& r) const {
          return r == *this;
        };
      };
    }
    template<typename T>
    using simple_storage = detail::data_storage<T, std::vector<T> >;
  }
}

#endif //ALPSCORE_GF_TENSORBASE_H
