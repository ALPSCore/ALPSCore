//
// Created by iskakoff on 26/10/17.
//

#ifndef GF2_TENSORVIEW_H
#define GF2_TENSORVIEW_H



#include <vector>
#include <array>
#include "DataStorage.h"
#include "View.h"

namespace alps {
  namespace gf {
    namespace detail {

       /**
        * @brief TensorView class. Provide interface for tensor to operate with raw data
        *
        * @author iskakoff
        */
      template<typename T>
      class DataView {
      public:
        /// Construct view of the whole DataStorage
        DataView(const DataStorage<T> & storage) : _data_slice(storage), _offset(0), _size(storage.size()) {}
        /// Construct subview of specified size for DataStorage starting from offset point
        DataView(const DataStorage<T> & storage, size_t size, size_t offset = 0) : _data_slice(storage), _offset(offset), _size(size) {}
        /// Move-construction of subview of specified size for another View starting from offset point
        DataView(DataView<T> && storage, size_t size, size_t offset) : _data_slice(storage._data_slice), _offset(offset + storage._offset), _size(size) {}
        /// Copy-construction of subview of specified size for another View starting from offset point
        DataView(const DataView<T> & storage, size_t size, size_t offset) : _data_slice(storage._data_slice), _offset(offset + storage._offset), _size(size) {}
        /// Create view for the raw buffer
        DataView(T*data, size_t size) : _data_slice(data, size), _offset(0), _size(size){}
        /// Copy constructor
        DataView(const DataView<T> & storage) : _data_slice(storage._data_slice), _offset(storage._offset), _size(storage._size) {}
        /// Move constructor
        DataView(DataView<T> && storage) : _data_slice(storage._data_slice), _offset(storage._offset), _size(storage._size) {}

        /// Copy assignment
        DataView<T>& operator=(const DataView<T>& rhs) {
          _data_slice = rhs._data_slice;
          _offset = rhs._offset;
          _size = rhs._size;
          return (*this);
        }
        /// Move assignment
        DataView<T>& operator=(DataView<T>&& rhs) {
          _data_slice = rhs._data_slice;
          _offset = rhs._offset;
          _size = rhs._size;
          return (*this);
        }

        /// @return reference to the data at point i
        T& data(size_t i) {return _data_slice.data(i + _offset);};
        /// @return const-reference to the data at point i
        const T& data(size_t i) const {return _data_slice.data(i + _offset);};
        /// @return buffer size
        size_t size() const {return _size;}
        /// @return offset from the buffer beginning
        size_t offset() const {return _offset;}
        /// @return pointer to the raw buffer
        T* data() {return _data_slice.data();}
        /// @return const pointer to the raw buffer
        const T* data() const {return _data_slice.data();}

        /// DataView comparison
        template<typename T2>
        bool operator==(const DataView<T2>& r) const {
          return size() == r.size() && std::equal(data(), data() + size(), data());
        }

        /// Comparison against DataStorage
        template<typename T2>
        bool operator==(const DataStorage<T2>& r) const {
          return size() == r.size() && std::equal(r.data().begin(), r.data().end(), data());
        }
      private:
        /// raw buffer storage
        View<T> _data_slice;
        /// data offset
        size_t _offset;
        /// data size
        size_t _size;

      };
    }
  }
}
#endif //GF2_TENSORVIEW_H
