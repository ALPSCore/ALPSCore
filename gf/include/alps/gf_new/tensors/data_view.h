/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef GF2_TENSORVIEW_H
#define GF2_TENSORVIEW_H



#include <vector>
#include <array>
#include <alps/gf_new/tensors/data_storage.h>
#include <alps/gf_new/tensors/view.h>

namespace alps {
  namespace gf {
    namespace detail {

       /**
        * @brief TensorView class. Provide interface for tensor to operate with raw data
        */
      template<typename T>
      class data_view {
      private:
        /// raw buffer storage
        view<T> data_slice_;
        /// data offset
        size_t offset_;
        /// data size
        size_t size_;
      public:
        /// Construct view of the whole DataStorage
        data_view(data_storage<T> & storage) : data_slice_(storage), offset_(0), size_(storage.size()) {}
        data_view(const data_storage<T> & storage) : data_slice_(storage), offset_(0), size_(storage.size()) {}
        /// Construct subview of specified size for DataStorage starting from offset point
        data_view(data_storage<T> & storage, size_t size, size_t offset = 0) : data_slice_(storage), offset_(offset), size_(size) {}
        data_view(const data_storage<T> & storage, size_t size, size_t offset = 0) : data_slice_(storage), offset_(offset), size_(size) {}
        /// Move-construction of subview of specified size for another View starting from offset point
        data_view(data_view<T> && storage, size_t size, size_t offset) : data_slice_(storage.data_slice_), offset_(offset + storage.offset_), size_(size) {}
        /// Copy-construction of subview of specified size for another View starting from offset point
        data_view(const data_view<T> & storage, size_t size, size_t offset) : data_slice_(storage.data_slice_), offset_(offset + storage.offset_), size_(size) {}
        /// Create view for the raw buffer
        data_view(T*data, size_t size) : data_slice_(data, size), offset_(0), size_(size){}
        /// Copy constructor
        data_view(const data_view<T> & storage) : data_slice_(storage.data_slice_), offset_(storage.offset_), size_(storage.size_) {}
        /// Move constructor
        data_view(data_view<T> && storage) : data_slice_(storage.data_slice_), offset_(storage.offset_), size_(storage.size_) {}

        /// Copy assignment
        data_view<T>& operator=(const data_view<T>& rhs) {
          data_slice_ = rhs.data_slice_;
          offset_ = rhs.offset_;
          size_ = rhs.size_;
          return (*this);
        }
        /// Move assignment
        data_view<T>& operator=(data_view<T>&& rhs) {
          data_slice_ = rhs.data_slice_;
          offset_ = rhs.offset_;
          size_ = rhs.size_;
          return (*this);
        }

        /// @return reference to the data at point i
        T& data(size_t i) {return data_slice_.data(i + offset_);};
        /// @return const-reference to the data at point i
        const T& data(size_t i) const {return data_slice_.data(i + offset_);};
        /// @return buffer size
        size_t size() const {return size_;}
        /// @return offset from the buffer beginning
        size_t offset() const {return offset_;}
        /// @return pointer to the raw buffer
        T* data() {return data_slice_.data();}
        /// @return const pointer to the raw buffer
        const T* data() const {return data_slice_.data();}

        /// DataView comparison
        template<typename T2>
        bool operator==(const data_view<T2>& r) const {
          return size() == r.size() && std::equal(data(), data() + size(), data());
        }

        /// Comparison against DataStorage
        template<typename T2>
        bool operator==(const data_storage<T2>& r) const {
          return size() == r.size() && std::equal(r.data().begin(), r.data().end(), data());
        }
      };
    }
  }
}
#endif //GF2_TENSORVIEW_H
