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
 * @brief TensorView class
 *
 * @author iskakoff
 */
template<typename T>
class DataView {
public:
  DataView(const DataStorage<T> & storage) : _data_slice(storage), _offset(0), _size(storage.size()) {}
  DataView(const DataStorage<T> & storage, size_t size, size_t offset = 0) : _data_slice(storage), _offset(offset), _size(size) {}
  DataView(DataView<T> && storage, size_t size, size_t offset) : _data_slice(storage._data_slice), _offset(offset + storage._offset), _size(size) {}
  DataView(const DataView<T> & storage, size_t size, size_t offset) : _data_slice(storage._data_slice), _offset(offset + storage._offset), _size(size) {}

  DataView(T*data, size_t size) : _data_slice(data, size), _offset(0), _size(size){}

  DataView(const DataView<T> & storage) : _data_slice(storage._data_slice), _offset(storage._offset), _size(storage._size) {}
  DataView(DataView<T> && storage) : _data_slice(storage._data_slice), _offset(storage._offset), _size(storage._size) {}

  DataView<T>& operator=(const DataView<T>& rhs) {
    _data_slice = rhs._data_slice;
    _offset = rhs._offset;
    _size = rhs._size;
    return (*this);
  }

  DataView<T>& operator=(DataView<T>&& rhs) {
    _data_slice = rhs._data_slice;
    _offset = rhs._offset;
    _size = rhs._size;
    return (*this);
  }

  T& data(size_t i) {return _data_slice.data(i + _offset);};
  const T& data(size_t i) const {return _data_slice.data(i + _offset);};

  size_t size() const {return _size;}

  size_t offset() const {return _offset;}
  T* data() {return _data_slice.data();}
private:

  View<T> _data_slice;
  size_t _offset;
  size_t _size;

};


template<typename T1, typename T2>
bool operator==(const DataView<T1>& l, const DataStorage<T2> r) {
  return l.size() == r.size() && std::equal(r.data().begin(), r.data().end(), l.data());
}

template<typename T1, typename T2>
bool operator==(const DataStorage<T2> l, const DataView<T1>& r) {
  return r == l;
};
}
}
}
#endif //GF2_TENSORVIEW_H
