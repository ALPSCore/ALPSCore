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
       * @brief View class. Store reference to the raw buffer
       *
       * @author iskakoff
       */
      template<typename T>
      class View {
      public:
        /**
         * @param data - pointer to the raw buffer
         * @param size - raw buffer size
         */
        View(T* data, size_t size) : _data(data), _size(size){}
        /**
         * Construct view on DataStorage object
         */
        View(const DataStorage<T>&storage) : _data(const_cast<T*>(storage.data().data())), _size(storage.data().size())  {}
        /// Copy constructor
        View(const View & view) : _data(view._data), _size(view._size) {}
        /// Move constructor
        View(View && view) : _data(view._data), _size(view._size) {}

        /// assignment operator
        View<T>& operator=(const View<T> & rhs) {
          _data = rhs._data;
          _size = rhs._size;
          return (*this);
        }

        /// move assignment operator
        View<T> operator=(View<T> && rhs) {
          _data = rhs._data;
          _size = rhs._size;
          return (*this);
        }

        /// @return reference to the data at i-th point
        inline T& data(size_t i) {return _data[i];};
        /// @return const reference to the data at i-th point
        inline const T& data(size_t i) const {return _data[i];};
        /// @return size of the raw buffer
        virtual size_t size() const {return _size;}
        /// @return raw buffer pointer
        inline T* data() {return _data;};
        /// @return const raw buffer pointer
        inline const T* data() const {return _data;};

      private:
        /// raw buffer pointer
        T* _data;
        /// raw buffer size
        size_t _size;
      };
    }
  }
}

#endif //GF2_VIEW_H
