/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2002-2003 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef ALPS_FIXED_CAPACITY_UNINITIALIZED_ARRAY_HPP
#define ALPS_FIXED_CAPACITY_UNINITIALIZED_ARRAY_HPP

#include <boost/config.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/alignment_traits.hpp>
#include <cstddef>

namespace alps {

// class template uninitialized_array ---------------------------------------//

template<class T, std::size_t N>
class uninitialized_array
{
public:
  // types:
  typedef std::size_t size_type;
  typedef T           value_type;
  typedef T&          reference;
  typedef const T&    const_reference;
  typedef T*          iterator;
  typedef const T*    const_iterator;

  BOOST_STATIC_CONSTANT(size_type, static_size = N);

  // compiler-generated constructors/destructor are fine

  // iterators:
  iterator begin() { return reinterpret_cast<iterator>(buffer_); }
  const_iterator begin() const {
    return reinterpret_cast<const_iterator>(buffer_);
  }
  iterator end() { return begin() + N; }
  const_iterator end() const { return begin() + N; }

  // capacity:
  static size_type size() { return N; }

  // element access:
  reference operator[](size_type i) { return *(begin() + i); }
  const_reference operator[](size_type i) const { return *(begin() + i); }
  
private:
  BOOST_STATIC_ASSERT(N > 0);

  union {
    char buffer_[N * sizeof(T)];
    typename boost::type_with_alignment<boost::alignment_of<T>::value>::type
      dummy_;
  };

}; // uninitialized_array

} // end namespace alps

#endif // ALPS_FIXED_CAPACITY_UNINITIALIZED_ARRAY_HPP
