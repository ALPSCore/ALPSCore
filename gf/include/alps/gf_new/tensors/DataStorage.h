//
// Created by iskakoff on 26/10/17.
//

#ifndef GF2_TENSORBASE_H
#define GF2_TENSORBASE_H


namespace alps {
  namespace gf {
    namespace detail {

      // Forward declarations
      template<typename T>
      class DataView;

/**
 * @brief TensorBase class
 *
 * @author iskakoff
 */
      template<typename T>
      class DataStorage {
      public:
        /// Create DataStorage from the view object by copying data into vector
        explicit DataStorage(const DataView<T> & view): _data(view.size()) {
          for(int i =0 ; i< view.size(); ++i) {
            _data[i] = view.data(i);
          }
        }
        /// Move-Create DataStorage from the view object by copying data into vector
        explicit DataStorage(DataView<T> && view): _data(view.size()) {
          for(int i =0 ; i< view.size(); ++i) {
            _data[i] = view.data(i);
          }
        }
        /// Create data storage from raw buffer by data copying
        DataStorage(T *data, size_t size): _data(size){
          for(int i =0 ; i< size; ++i) {
            _data[i] = data[i];
          }
        }
        /// Create empty storage of size %size%
        explicit DataStorage(size_t size): _data(size, T(0)) {}

        /// @return reference to the data at point i
        inline T& data(size_t i) {return _data[i];};
        /// @return const-reference to the data at point i
        inline const T& data(size_t i) const {return _data[i];};
        /// @return data size
        virtual size_t size() const {return _data.size();}
        /// @return const-reference to stored vector
        const std::vector<T>& data() const {return _data;}
        /// @return reference to stored vector
        std::vector<T>& data() {return _data;}

        /**
         * Data Storage comparison. Two data storages are equal if they have the same size
         * and all elements are equal to each other
         *
         * @tparam T2 - datatype of rhs storage
         * @param r   - rhs storage
         */
        template<typename T2>
        bool operator==(const DataStorage<T2> &r) const {
          return r.size() == size() && std::equal(r.data().begin(), r.data().end(), _data.begin());
        }

        /// Comparison against DataView
        template<typename T2>
        bool operator==(const DataView<T2>& r) const {
          return r == *this;
        };

      private:
        /// internal data storage
        std::vector<T> _data;

      };
    }
  }
}

#endif //GF2_TENSORBASE_H
