/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_GF_VIEW_H
#define ALPSCORE_GF_VIEW_H


#include <alps/gf_new/tensors/data_storage.h>

namespace alps {
  namespace gf {
    namespace detail {

      /**
       * @brief View class. Store reference to the raw buffer
       *
       * @author iskakoff
       */
      template<typename T>
      class view {
      private:
        /// raw buffer pointer
        T* data_;
        /// raw buffer size
        size_t size_;
      public:
        /**
         * @param data - pointer to the raw buffer
         * @param size - raw buffer size
         */
        view(T* data, size_t size) : data_(data), size_(size){}
        /**
         * Construct view on DataStorage object
         */
        view(const data_storage<T>&storage) : data_(storage.data().data()), size_(storage.data().size())  {}
        view(data_storage<T>&storage) : data_(storage.data().data()), size_(storage.data().size())  {}
        /// Copy constructor
        view(const view & view) : data_(view.data_), size_(view.size_) {}
        /// Move constructor
        view(view && view) : data_(view.data_), size_(view.size_) {}

        /// assignment operator
        view<T>& operator=(const view<T> & rhs) {
          data_ = rhs.data_;
          size_ = rhs.size_;
          return (*this);
        }

        /// move assignment operator
        view<T> operator=(view<T> && rhs) {
          data_ = rhs.data_;
          size_ = rhs.size_;
          return (*this);
        }

        /// @return reference to the data at i-th point
        inline T& data(size_t i) {return data_[i];};
        /// @return const reference to the data at i-th point
        inline const T& data(size_t i) const {return data_[i];};
        /// @return size of the raw buffer
        size_t size() const {return size_;}
        /// @return raw buffer pointer
        inline T* data() {return data_;};
        /// @return const raw buffer pointer
        inline const T* data() const {return data_;};
      };
    }
  }
}

#endif //ALPSCORE_GF_VIEW_H
