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

/// \file matrix_as_vector.hpp
/// \brief using matrices as vectors in generic algorithms
/// 
/// This header contains a wrapper class that allows to use a matrix
/// instead of a vector in a generic algorithm.  As long as the
/// algorithm does nothing more than to get a vector element, multiply
/// it with a factor and add it to another vector element, the \c
/// matrix_as_vector class can record these actions into a matrix.


#ifndef ALPS_NUMERIC_MATRIX_AS_VECTOR_HPP
#define ALPS_NUMERIC_MATRIX_AS_VECTOR_HPP

#include <alps/config.h>

namespace alps { namespace numeric {

namespace detail {

/// a vector element, scaled by an arbitrary factor
///
/// used as \c value_type of the \c matrix_as_vector class
/// \param M the matrix type
template <class M>
class element_proxy {
public:
/// the matrix type
  typedef M matrix_type;
/// the scalar type, to store scale factors
  typedef typename matrix_type::value_type value_type;
/// the type used to store indices
  typedef std::size_t size_type;

/// creates a proxy refering to a vector element
///
/// the scale factor is initialized to 1 and the proxy refers to the \a i -th element of the vector
/// \param m the matrix
/// \param i the index of the vector element
  element_proxy(matrix_type& m, size_type i) : matrix_(m), index_(i), val_(1.) {}
/// the index of the vector element the object refers to
  size_type index() const { return index_;}
/// the scalar by which the matrix element has been multiplied
  value_type value() const { return val_;}
/// multiplies the scale factor by the argument \a x
  const element_proxy<M>& operator*=(value_type x) { val_*=x; return *this;}
/// \brief stores a vector operation in matrix form
///
/// this allows operations like a[i] = b*c[j] to be recorded in matrix form. 
/// The \c value() is stored at location (i,j) into the matrix.
  template <class MM>
  void operator=(const element_proxy<MM>& x) { matrix_(index(),x.index())=x.value();}
/// \brief adds a vector operation in matrix form
///
/// this allows operations like a[i] += b*c[j] to be recorded in matrix form. 
/// The \c value() is added to the location (i,j) into the matrix.
  template <class MM>
  void operator+=(const element_proxy<MM>& x) { matrix_(index(),x.index())+=x.value();}

/// \brief subtracts a vector operation in matrix form
///
/// this allows operations like a[i] -= b*c[j] to be recorded in matrix form. 
/// The \c value() is subtracted from the location (i,j) into the matrix.
  template <class MM>
  void operator-=(const element_proxy<MM>& x) { matrix_(index(),x.index())-=x.value();}
private:
  matrix_type& matrix_;
  size_type index_;
  value_type val_;
  bool is_const_;
};

}

/// a vector that stores the result of simple operations in matrix form
///
/// This class allows to use a matrix instead of a vector in some generic algorithms. 
/// For example, vector operations such as a[j] = b * c[i] result in the value b being stored
/// at location (i,j) in the matrix
/// @param M the matrix type. It needs to support element access in the form  m(i,j), for objects m of type M.
template <class M>
class matrix_as_vector {
public:
/// the matrix type
  typedef M matrix_type;
/// the value type of the matrix
  typedef typename matrix_type::value_type value_type;
/// the type to store vector and matrix inidices
  typedef std::size_t size_type;

/// @param m the matrix that will be written into. It is stored by reference.
  matrix_as_vector(matrix_type& m) : matrix_(m) {}
/// returns the underlying matrix
  matrix_type& matrix() { return matrix_;}
/// returns the underlying matrix
  const matrix_type& matrix() const { return matrix_;}
/// \param i the index into the vector
/// \returns an element_proxy referring to the \a i -th vector element
  detail::element_proxy<M> operator[](size_type i) { return detail::element_proxy<M>(matrix(),i);}
/// \param i the index into the vector
/// \returns an element_proxy referring to the \a i -th vector element
  detail::element_proxy<const M> operator[](size_type i) const { return detail::element_proxy<const M>(matrix(),i);}
private:
  matrix_type& matrix_;
};

} } // end namespace alps::numeric

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {  namespace numeric  {namespace detail {
#endif

/// \brief multiplication of an element_proxy with a scalar
///
/// \param x the vector element proxy
/// \param y the scalar
/// \returns the result of x *= y
template <class M, class T>
alps::numeric::detail::element_proxy<M> operator*(alps::numeric::
detail::element_proxy<M> x, T y)
{
  return x *= y;
}

/// \brief multiplication of an element_proxy with a scalar
///
/// \param x the scalar
/// \param y the vector element proxy
/// \returns the result of y *= x
template <class M, class T>
alps::numeric::detail::element_proxy<M> operator*(T x, alps::numeric::detail::element_proxy<M> y)
{
  return y *= x;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} } }
#endif

#endif // ALPS_NUMERIC_MATRIX_AS_VECTOR_HPP
