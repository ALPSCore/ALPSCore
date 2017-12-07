//
// Created by iskakoff on 26/10/17.
//

#ifndef GF2_TENSORBASE_H
#define GF2_TENSORBASE_H


namespace alps {
  namespace gf {
    namespace detail {

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

  explicit DataStorage(const DataView<T> & view): _data(view.size()) {
    for(int i =0 ; i< view.size(); ++i) {
      _data[i] = view.data(i);
    }
  }

  explicit DataStorage(DataView<T> && view): _data(view.size()) {
    for(int i =0 ; i< view.size(); ++i) {
      _data[i] = view.data(i);
    }
  }

  DataStorage(T *data, size_t size): _data(size){
    for(int i =0 ; i< size; ++i) {
      _data[i] = data[i];
    }
  }

  explicit DataStorage(size_t size): _data(size) {}

  inline T& data(size_t i) {return _data[i];};
  inline const T& data(size_t i) const {return _data[i];};
  virtual size_t size() const {return _data.size();}

  const std::vector<T>& data() const {return _data;}
  std::vector<T>& data() {return _data;}

  template<typename T2>
  bool operator==(const DataStorage<T2> &r) {
    return r.size() == size() && std::equal(r.data().begin(), r.data().end(), _data.begin());
  }
private:

  std::vector<T> _data;

};
}
}
}

#endif //GF2_TENSORBASE_H
