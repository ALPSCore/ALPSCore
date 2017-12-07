//
// Created by iskakoff on 03/11/17.
//

#ifndef GF2_VIEW_H
#define GF2_VIEW_H


#include "DataStorage.h"

namespace alps {
  namespace gf {
    namespace detail {

/**
 * @brief View class
 *
 * @author iskakoff
 */
template<typename T>
class View {
public:
  View(T* data, size_t size) : _data(data), _size(size){}
  View(const DataStorage<T>&storage) : _data(const_cast<T*>(storage.data().data())), _size(storage.data().size())  {}
  View(const View & view) : _data(view._data), _size(view._size) {}
  View(View && view) : _data(view._data), _size(view._size) {}

  View<T> operator=(const View<T> & rhs) {
    _data = rhs._data;
    _size = rhs._size;
    return (*this);
  }

  View<T> operator=(View<T> && rhs) {
    _data = rhs._data;
    _size = rhs._size;
    return (*this);
  }

  inline T& data(size_t i) {return _data[i];};
  inline const T& data(size_t i) const {return _data[i];};
  virtual size_t size() const {return _size;}
  inline T* data() {return _data;};

private:
  T* _data;
  size_t _size;
};
}
}
}

#endif //GF2_VIEW_H
