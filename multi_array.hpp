/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2005 by Matthias Troyer <troyer@comp-phys.org>
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

/// \addtogroup alps
/// @{

/// \file multi_array.hpp
/// \brief extensions to boost::multi_array
///
/// This header defines some I/O extensions to boost::multi_array and fixes a problem with gcc-3.1 when
/// alps::multi_array and alps::serialization are used together

#ifndef ALPS_MULTI_ARRAY_H
#define ALPS_MULTI_ARRAY_H
/// @}


//=======================================================================
// This file defines extensions to boost::multi_array
//=======================================================================


#include <alps/config.h>
#define access alps_multiarray_access
#include <boost/multi_array.hpp>
#undef access
#include <boost/throw_exception.hpp>
#include <cmath>
#include <stdexcept>

#ifndef ALPS_WITHOUT_OSIRIS
# include <alps/osiris.h>
#endif

//
// OSIRIS support
//

#ifndef ALPS_WITHOUT_OSIRIS

namespace alps {

namespace detail {

/// a helper class to (de)serialize a multi_array
template <bool OPTIMIZED> struct MultiArrayHelper {};

/// a helper class to (de)serialize a multi_array of non-POD types
template <> struct MultiArrayHelper<false>
{
  /// \brief read the mutli-array from a dump
  ///
  /// implemented for non-POD data by iterating over the multi_array elements
  template <class T, std::size_t NumDims, class ALLOCATOR>
  static void read(IDump& dump, boost::multi_array<T, NumDims, ALLOCATOR>& x) 
  {
    for (T* p = x.data(); p != x.data() + x.num_elements(); ++p)
      dump >> *p;
  }

  /// \brief write the mutli-array to a dump
  ///
  /// implemented for non-POD data by iterating over the multi_array elements
  template <class T, std::size_t NumDims, class ALLOCATOR>
  static void write(ODump& dump,
                    const boost::multi_array<T, NumDims, ALLOCATOR>& x) 
  {
    for (T* p = x.data(); p != x.data() + x.num_elements(); ++p)
      dump << *p;
  }
};

/// a helper class to (de)serialize a multi_array of POD types
template <> struct MultiArrayHelper<true>
{
  /// \brief read the mutli-array from a dump
  ///
  /// implemented for POD data by calling read_array
  template <class T, std::size_t NumDims, class ALLOCATOR>
  static void read(IDump& dump, boost::multi_array<T, NumDims, ALLOCATOR>& x) 
  {
    dump.read_array(x.num_elements(), x.data());
  }
  
  /// \brief write the mutli-array to a dump
  ///
  /// implemented for non-POD data by calling write_array
  template <class T, std::size_t NumDims, class ALLOCATOR>
  static void write(ODump& dump,
                    const boost::multi_array<T, NumDims, ALLOCATOR>& x) 
  {
    dump.write_array(x.num_elements(), x.data());
  }
};

} // namespace detail

} // namespace alps

#endif

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif



#ifndef ALPS_WITHOUT_OSIRIS

/// \brief ALPS de-serialization support for boost::multi_array
template <class T, std::size_t NumDims, class Allocator>
alps::IDump& operator>>(alps::IDump& dump, boost::multi_array<T, NumDims, Allocator>& x)
{
  std::vector<uint32_t> ex;
  dump >> ex;
  if(ex.size() != NumDims)
    boost::throw_exception(std::runtime_error("Number of dimensions does not agree in reading multi_array"));
  x = boost::multi_array<T, NumDims, Allocator>(ex);
  alps::detail::MultiArrayHelper<alps::detail::TypeDumpTraits<T>::hasArrayFunction>::read(dump, x);
  return dump;
}

/// \brief ALPS serialization support for boost::multi_array
template <class T, std::size_t NumDims, class Allocator>
alps::ODump& operator<<(alps::ODump& dump, const boost::multi_array<T, NumDims, Allocator>& x)
{
  std::vector<uint32_t> ex(x.shape(), x.shape() + x.num_dimensions());
  dump << ex;
  alps::detail::MultiArrayHelper<alps::detail::TypeDumpTraits<T>::hasArrayFunction>::write(dump, x);
  return dump;
}          


#endif // !ALPS_WITHOUT_OSIRIS

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace boost
#endif

/// \brief writes a two-dimensional boost::multi_array to an output stream
template <class T, class Allocator>
std::ostream& operator<<(std::ostream& out, const boost::multi_array<T, 2, Allocator>& x)
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

/// \brief writes a four-dimensional boost::multi_array to an output stream
template <class T, class Allocator>
std::ostream& operator<<(std::ostream& out, const boost::multi_array<T, 4, Allocator>& x)
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

#endif // ALPS_MULTI_ARRAY_H
