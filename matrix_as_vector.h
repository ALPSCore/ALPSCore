/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1999-2005 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>
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

#ifndef ALPS_MATRIX_AS_VECTOR_H
#define ALPS_MATRIX_AS_VECTOR_H

#include <alps/config.h>

namespace alps {

namespace detail {

template <class M>
class element_proxy {
public:
  typedef M matrix_type;
  typedef typename matrix_type::value_type value_type;
  typedef std::size_t size_type;

  element_proxy(matrix_type& m, size_type i) : matrix_(m), index_(i), val_(1.) {}
  size_type index() const { return index_;}
  value_type value() const { return val_;}
  const element_proxy<M>& operator*=(value_type x) { val_*=x; return *this;}
  
  template <class MM>
  void operator=(const element_proxy<MM>& x) { matrix_(index(),x.index())=x.value();}
  
  template <class MM>
  void operator+=(const element_proxy<MM>& x) { matrix_(index(),x.index())+=x.value();}

  template <class MM>
  void operator-=(const element_proxy<MM>& x) { matrix_(index(),x.index())-=x.value();}
private:
  matrix_type& matrix_;
  size_type index_;
  value_type val_;
  bool is_const_;
};

}

template <class M>
class matrix_as_vector {
public:
  typedef M matrix_type;
  typedef typename matrix_type::value_type value_type;
  typedef std::size_t size_type;

  matrix_as_vector(matrix_type& m) : matrix_(m) {}
  matrix_type& matrix() { return matrix_;}
  const matrix_type& matrix() const { return matrix_;}
  detail::element_proxy<M> operator[](size_type i) { return detail::element_proxy<M>(matrix(),i);}
  detail::element_proxy<const M> operator[](size_type i) const { return detail::element_proxy<const M>(matrix(),i);}
private:
  matrix_type& matrix_;
};

} // end namespace

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps { namespace detail {
#endif

template <class M, class T>
alps::detail::element_proxy<M> operator*(alps::detail::element_proxy<M> x, T y)
{
  return x *= y;
}

template <class M, class T>
alps::detail::element_proxy<M> operator*(T x, alps::detail::element_proxy<M> y)
{
  return y *= x;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} }
#endif

#endif // ALPS_MATRIX_AS_VECTOR_H
