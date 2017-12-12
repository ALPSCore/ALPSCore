/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef GF2_TENSORBASE_H
#define GF2_TENSORBASE_H


namespace alps {
  namespace gf {
    namespace detail {

      // Forward declarations
      template<typename T>
      class data_view;

      /**
       * @brief TensorBase class implements basic multi-dimensional operations.
       * All arithmetic operations are performed by Eigen library
       */
      template<typename T>
      class data_storage {
      private:
        /// internal data storage
        std::vector<T> data_;
      public:
        /// Create DataStorage from the view object by copying data into vector
        explicit data_storage(const data_view<T> & view): data_(view.size()) {
          for(int i =0 ; i< view.size(); ++i) {
            data_[i] = view.data(i);
          }
        }
        /// Move-Create DataStorage from the view object by copying data into vector
        explicit data_storage(data_view<T> && view): data_(view.size()) {
          for(int i =0 ; i< view.size(); ++i) {
            data_[i] = view.data(i);
          }
        }
        /// Create data storage from raw buffer by data copying
        data_storage(const T *data, size_t size): data_(data, data+size){}
        /// Create empty storage of size %size%
        explicit data_storage(size_t size): data_(size, T(0)) {}

        /// @return reference to the data at point i
        inline T& data(size_t i) {return data_[i];};
        /// @return const-reference to the data at point i
        inline const T& data(size_t i) const {return data_[i];};
        /// @return data size
        size_t size() const {return data_.size();}
        /// @return const-reference to stored vector
        const std::vector<T>& data() const {return data_;}
        /// @return reference to stored vector
        std::vector<T>& data() {return data_;}

        /**
         * Data Storage comparison. Two data storages are equal if they have the same size
         * and all elements are equal to each other
         *
         * @tparam T2 - datatype of rhs storage
         * @param r   - rhs storage
         */
        template<typename T2>
        bool operator==(const data_storage<T2> &r) const {
          return r.size() == size() && std::equal(r.data().begin(), r.data().end(), data_.begin());
        }

        /// Comparison against DataView
        template<typename T2>
        bool operator==(const data_view<T2>& r) const {
          return r == *this;
        };
      };
    }
  }
}

#endif //GF2_TENSORBASE_H
