#ifndef ALPS_VECTOR_VALARRAY_CONVERSION
#define ALPS_VECTOR_VALARRAY_CONVERSION

#include <vector>
#include <valarray>
#include <algorithm>


namespace alps {
  namespace numeric {

    template<class T>
    void valarray2vector(std::valarray<T> const & from, std::vector<T> & to)
    {
      to.clear();
      to.reserve(from.size());
      std::copy(&from[0],&from[from.size()],std::back_inserter(to));
    }  

    template<class T>
    void vector2valarray(std::vector<T> const & from, std::valarray<T> & to)
    {
      to.resize(from.size());
      std::copy(from.begin(),from.end(),&to[0]);
    }

  }
}

#endif
