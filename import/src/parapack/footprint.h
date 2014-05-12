/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2010 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef PARAPACK_FOOTPRINT_H
#define PARAPACK_FOOTPRINT_H

#include <alps/config.h>
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

#include <string>
#include <vector>

namespace alps {

template<typename T>
std::size_t footprint(T const& t, typename boost::enable_if<boost::is_pod<T> >::type* = 0) {
  return sizeof(T);
}

template<typename T>
std::size_t footprint(T const& t, typename boost::disable_if<boost::is_pod<T> >::type* = 0) {
  return t.footprint();
}

template<typename T>
std::size_t footprint(std::vector<T> const& v) {
  return sizeof(std::vector<T>) + sizeof(T) * v.capacity();
}

template<typename C, typename T, typename A>
std::size_t footprint(std::basic_string<C, T, A> const& v) {
  return sizeof(std::basic_string<C, T, A>) + sizeof(C) * v.capacity();
}

} // end namespace alps

#endif // PARAPACK_FOOTPRINT_H
