/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003 by Matthias Troyer <troyer@comp-phys.org>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/* $Id$ */

#ifndef ALPS_MULTI_ARRAY_H
#define ALPS_MULTI_ARRAY_H

//=======================================================================
// This file defines extensions to boost::multi_array
//=======================================================================

#include <alps/config.h>
#include <boost/multi_array.hpp>
#include <boost/throw_exception.hpp>
#include <cmath>
#include <stdexcept>

#ifndef ALPS_WITHOUT_OSIRIS
# include <alps/osiris.h>
#endif

namespace alps {

template <class T, std::size_t NumDims, class Allocator= std::allocator<T> >
class multi_array : public boost::multi_array<T,NumDims,Allocator>
{
public:
  typedef boost::multi_array<T,NumDims,Allocator> super_type;

  typedef typename super_type::value_type value_type;
  typedef typename super_type::reference reference;
  typedef typename super_type::const_reference const_reference;
  typedef typename super_type::reverse_iterator reverse_iterator;
  typedef typename super_type::const_reverse_iterator const_reverse_iterator;
  typedef typename super_type::element element;
  typedef typename super_type::size_type size_type;
  typedef typename super_type::difference_type difference_type;
  typedef typename super_type::index index;
  typedef typename super_type::extent_range extent_range;

  multi_array(const super_type& rhs) : super_type(rhs) {}
  
  // Duplicating constructors
  
  template <class ExtentList>
  explicit multi_array(ExtentList const& extents) 
   : super_type(extents) {}

  template <class ExtentList>
  explicit multi_array(ExtentList const& extents,const boost::general_storage_order<NumDims>& so) 
   : super_type(extents,so) {}

  template <class ExtentList>
  explicit multi_array(ExtentList const& extents,
                       const boost::general_storage_order<NumDims>& so,
                       Allocator const& alloc) 
   : super_type(extents,so,alloc) {}


  explicit multi_array(const boost::detail::multi_array::extent_gen<NumDims>& ranges) 
   : super_type(ranges) {}


  explicit multi_array(const boost::detail::multi_array::extent_gen<NumDims>& ranges,
                       const boost::general_storage_order<NumDims>& so) 
   : super_type(ranges,so) {}


  explicit multi_array(const boost::detail::multi_array::extent_gen<NumDims>& ranges,
                       const boost::general_storage_order<NumDims>& so, Allocator const& alloc) 
   : super_type(ranges,so,alloc) {}

  template <typename OPtr>
  multi_array(const boost::detail::multi_array::const_sub_array<T,NumDims,OPtr>& rhs) 
   : super_type(rhs) {}

  // For some reason, gcc 2.95.2 doesn't pick the above template
  // member function when passed a subarray, so i was forced to
  // duplicate the functionality here...
  multi_array(const boost::detail::multi_array::sub_array<T,NumDims>& rhs) 
   : super_type(rhs) {}
   
   
  // my extensions

  multi_array() {}
  
  typedef T* iterator;
  typedef const T* const_iterator;
  
  iterator begin() { return super_type::data();}
  const_iterator begin() const { return super_type::data();}
  iterator end() { return super_type::data()+super_type::num_elements();}
  const_iterator end() const { return super_type::data()+super_type::num_elements();}

  template <class X, class Alloc>
  const multi_array& operator=(const alps::multi_array<X,NumDims,Alloc>& x) 
  {
    super_type::operator=(x);
    return *this;
  }

  const multi_array& operator=(T x) 
  {
    std::fill(begin(),end(),x);
    return *this;
  }
  
  multi_array operator-() const {
    multi_array res(*this);
    std::transform(res.begin(),res.end(),res.begin(),std::negate<T>());
    return res;
  }
 
const multi_array& operator+=(const multi_array& x)
{
  std::transform(begin(),end(),x.begin(),begin(),std::plus<T>());
  return *this;
}

const multi_array& operator-=(const multi_array& x)
{
  std::transform(begin(),end(),x.begin(),begin(),std::minus<T>());
  return *this;
}

const multi_array& operator*=(const multi_array& x)
{
  std::transform(begin(),end(),x.begin(),begin(),std::multiplies<T>());
  return *this;
}

const multi_array& operator/=(const multi_array& x)
{
  std::transform(begin(),end(),x.begin(),begin(),std::divides<T>());
  return *this;
}

const multi_array& operator+=(const T& x)
{
  std::transform(begin(),end(),begin(),boost::bind2nd(std::plus<T>(),x));
  return *this;
}

const multi_array& operator-=(const T& x)
{
  std::transform(begin(),end(),begin(),boost::bind2nd(std::minus<T>(),x));
  return *this;
}

const multi_array& operator*=(const T& x)
{
  std::transform(begin(),end(),begin(),boost::bind2nd(std::multiplies<T>(),x));
  return *this;
}

const multi_array& operator/=(const T& x)
{
  std::transform(begin(),end(),begin(),boost::bind2nd(std::divides<T>(),x));
  return *this;
}

};

template <class T, std::size_t NumDims, class Allocator>
multi_array<T,NumDims,Allocator> operator/(const multi_array<T,NumDims,Allocator>& x, T y)
{
  multi_array<T,NumDims,Allocator> res(x);
  std::transform(res.begin(),res.end(),res.begin(),boost::bind2nd(std::divides<T>(),y));
  return res;
}

template <class T, std::size_t NumDims, class Allocator>
multi_array<T,NumDims,Allocator> operator-(const multi_array<T,NumDims,Allocator>& x, T y)
{
  multi_array<T,NumDims,Allocator> res(x);
  std::transform(res.begin(),res.end(),res.begin(),boost::bind2nd(std::minus<T>(),y));
  return res;
}

template <class T, std::size_t NumDims, class Allocator>
multi_array<T,NumDims,Allocator> operator+(const multi_array<T,NumDims,Allocator>& x, T y)
{
  multi_array<T,NumDims,Allocator> res(x);
  std::transform(res.begin(),res.end(),res.begin(),boost::bind2nd(std::plus<T>(),y));
  return res;
}

template <class T, std::size_t NumDims, class Allocator>
multi_array<T,NumDims,Allocator> operator-(T y, const multi_array<T,NumDims,Allocator>& x)
{
  multi_array<T,NumDims,Allocator> res(x);
  std::transform(res.begin(),res.end(),res.begin(),boost::bind1st(std::minus<T>(),y));
  return res;
}

template <class T, std::size_t NumDims, class Allocator>
multi_array<T,NumDims,Allocator> operator+(T y, const multi_array<T,NumDims,Allocator>& x)
{
  multi_array<T,NumDims,Allocator> res(x);
  std::transform(res.begin(),res.end(),res.begin(),boost::bind1st(std::plus<T>(),y));
  return res;
}

template <class T, std::size_t NumDims, class Allocator>
multi_array<T,NumDims,Allocator> operator*(const multi_array<T,NumDims,Allocator>& x, T y)
{
  multi_array<T,NumDims,Allocator> res(x);
  std::transform(res.begin(),res.end(),res.begin(),boost::bind2nd(std::multiplies<T>(),y));
  return res;
}
  
template <class T, std::size_t NumDims, class Allocator>
multi_array<T,NumDims,Allocator> operator*(T x, const multi_array<T,NumDims,Allocator>& y)
{
  multi_array<T,NumDims,Allocator> res(y);
  std::transform(res.begin(),res.end(),res.begin(),boost::bind1st(std::multiplies<T>(),x));
  return res;
}


template <class T, std::size_t NumDims, class Allocator>
multi_array<T,NumDims,Allocator> operator*(const multi_array<T,NumDims,Allocator>& x, const multi_array<T,NumDims,Allocator>& y)
{
  multi_array<T,NumDims,Allocator> res(x);
  std::transform(res.begin(),res.end(),y.begin(),res.begin(),std::multiplies<T>());
  return res;
}

template <class T, std::size_t NumDims, class Allocator>
multi_array<T,NumDims,Allocator> operator/(const multi_array<T,NumDims,Allocator>& x, const multi_array<T,NumDims,Allocator>& y)
{
  multi_array<T,NumDims,Allocator> res(x);
  std::transform(res.begin(),res.end(),y.begin(),res.begin(),std::divides<T>());
  return res;
}

template <class T, std::size_t NumDims, class Allocator>
multi_array<T,NumDims,Allocator> operator-(const multi_array<T,NumDims,Allocator>& x, const multi_array<T,NumDims,Allocator>& y)
{
  multi_array<T,NumDims,Allocator> res(x);
  std::transform(res.begin(),res.end(),y.begin(),res.begin(),std::minus<T>());
  return res;
}

template <class T, std::size_t NumDims, class Allocator>
multi_array<T,NumDims,Allocator> operator+(const multi_array<T,NumDims,Allocator>& x, const multi_array<T,NumDims,Allocator>& y)
{
  multi_array<T,NumDims,Allocator> res(x);
  std::transform(res.begin(),res.end(),y.begin(),res.begin(),std::plus<T>());
  return res;
}

namespace {
   template <class T> struct do_sqrt {
    T operator()(const T& x) const { using std::sqrt; return sqrt(x);}
  };
}

using std::sqrt;

template <class T, std::size_t NumDims, class Allocator>
alps::multi_array<T,NumDims,Allocator> sqrt(const alps::multi_array<T,NumDims,Allocator>& x)
{
  multi_array<T,NumDims,Allocator> res(x);
  std::transform(res.begin(),res.end(),res.begin(),do_sqrt<T>());
  return res;
}

} // end namespace


//
// OSIRIS support
//

#ifndef ALPS_WITHOUT_OSIRIS

namespace alps {

namespace detail {

template <bool OPTIMIZED> struct MultiArrayHelper {};

template <> struct MultiArrayHelper<false>
{
  template <class T, std::size_t NumDims, class ALLOCATOR>
  static void read(IDump& dump, boost::multi_array<T, NumDims, ALLOCATOR>& x) 
  {
    for (T* p = x.data(); p != x.data() + x.num_elements(); ++p)
      dump >> *p;
  }
  template <class T, std::size_t NumDims, class ALLOCATOR>
  static void write(ODump& dump,
                    const boost::multi_array<T, NumDims, ALLOCATOR>& x) 
  {
    for (T* p = x.data(); p != x.data() + x.num_elements(); ++p)
      dump << *p;
  }
};

template <> struct MultiArrayHelper<true>
{
  template <class T, std::size_t NumDims, class ALLOCATOR>
  static void read(IDump& dump, boost::multi_array<T, NumDims, ALLOCATOR>& x) 
  {
    dump.read_array(x.num_elements(), x.data());
  }
  
  template <class T, std::size_t NumDims, class ALLOCATOR>
  static void write(ODump& dump,
                    const boost::multi_array<T, NumDims, ALLOCATOR>& x) 
  {
    dump.write_array(x.num_elements(), x.data());
  }
};

} // namespace detail

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class T, std::size_t NumDims, class Allocator>
inline alps::IDump& operator>>(alps::IDump& dump, boost::multi_array<T, NumDims, Allocator>& x)
{
  std::vector<uint32_t> ex;
  dump >> ex;
  if(ex.size() != NumDims)
    boost::throw_exception(std::runtime_error("Number of dimensions does not agree in reading multi_array"));
  x = boost::multi_array<T, NumDims, Allocator>(ex);
  alps::detail::MultiArrayHelper<alps::detail::TypeDumpTraits<T>::hasArrayFunction>::read(dump, x);
  return dump;
}

/// serialize a std::vector container
template <class T, std::size_t NumDims, class Allocator>
inline alps::ODump& operator<<(alps::ODump& dump, const boost::multi_array<T, NumDims, Allocator>& x)
{
  std::vector<uint32_t> ex(x.shape(), x.shape() + x.num_dimensions());
  dump << ex;
  alps::detail::MultiArrayHelper<alps::detail::TypeDumpTraits<T>::hasArrayFunction>::write(dump, x);
  return dump;
}          

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // !ALPS_WITHOUT_OSIRIS

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace boost {
#endif

/// write a boost::multi_array 2-d array
template <class T, class Allocator>
inline std::ostream& operator<<(std::ostream& out, const boost::multi_array<T, 2, Allocator>& x)
{
  std::vector<uint32_t> ex(x.shape(), x.shape() + x.num_dimensions());
  out << "{";
  for (int i=0;i<ex[0];++i) {
    out << "{";
    for (int j=0;j<ex[1];++j) {
      out << x[i][j];
      if (j!=ex[1]-1)
        out << ", ";
    }
    out << "}";
    if (i!=ex[0]-1)
      out << ",\n";
  }
  out << "};";
  return out;
}          

/// write a boost::multi_array 4-d array
template <class T, class Allocator>
inline std::ostream& operator<<(std::ostream& out, const boost::multi_array<T, 4, Allocator>& x)
{
  std::vector<uint32_t> ex(x.shape(), x.shape() + x.num_dimensions());
  out << "{";
  for (int i=0;i<ex[0];++i) {
    out << "{";
    for (int j=0;j<ex[1];++j) {
      out << "{";
      for (int k=0;k<ex[2];++k) {
        out << "{";
        for (int l=0;l<ex[3];++l) {
          out << x[i][j][k][l];
          if (l!=ex[3]-1)
            out << ", ";
          }
        out << "}";
        if (k!=ex[2]-1)
          out << ", ";
      }
      out << "}";
      if (j!=ex[1]-1)
        out << ",\n";
    }
    out << "}";
    if (i!=ex[0]-1)
      out << ",\n";
  }
  out << "};";
  return out;
}          

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace boost
#endif

#endif // ALPS_MULTI_ARRAY_H
