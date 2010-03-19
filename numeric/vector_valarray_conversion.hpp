#ifndef ALPS_VECTOR_VALARRAY_CONVERSION
#define ALPS_VECTOR_VALARRAY_CONVERSION

#include <vector>
#include <valarray>
#include <algorithm>


namespace alps {
  namespace numeric {

    template<class T>
    std::vector<T> valarray2vector(std::valarray<T> const & from)
    {
      std::vector<T> to;
      to.reserve(from.size());
	  std::copy(&const_cast<std::valarray<T>&>(from)[0],&const_cast<std::valarray<T>&>(from)[0]+from.size(),std::back_inserter(to));
      return to;
    }

    template<class T>
    std::valarray<T> vector2valarray(std::vector<T> const & from)
    {
      std::valarray<T> to(from.size());
      std::copy(from.begin(),from.end(),&to[0]);
      return to;
    }

  }
}

#endif
