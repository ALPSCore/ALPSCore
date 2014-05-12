/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2013 by Andreas Hehn <hehn@phys.ethz.ch>                          *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef ALPS_NUMERIC_MATRIX_MATRIX_CONCEPT_ARCHETYPE_HPP
#define ALPS_NUMERIC_MATRIX_MATRIX_CONCEPT_ARCHETYPE_HPP

#include <boost/concept_archetypes.hpp>

namespace alps {
namespace numeric {
namespace concepts {


template <typename T>
class matrix_archetype
{
  /**
     \brief Class docs?

   **/
private:
    typedef matrix_archetype self;
public:
    typedef T               value_type;      //< The type of the coefficients stored in the matrix
    typedef std::size_t     size_type;       //< An unsigned integral type used to represent the size (num_rows, num_cols) of the matrix
    typedef std::ptrdiff_t  difference_type; //< A signed integral type used to represent the distance between two of the container's iterators

    typedef mutable_random_access_iterator_archetype<T> row_element_iterator;            //< An iterator that iterates over the elements of a row
    typedef random_access_iterator_archetype<T>         const_row_element_iterator;      //< A constant version of the row_element_iteratior
    typedef mutable_random_access_iterator_archetype<T> col_element_iterator;            //< An iterator that iterates over the elements of a column
    typedef random_access_iterator_archetype<T>         const_col_element_iterator;      //< A constant version of the col_element_iterator
    // TODO more typedefs

    /**
      * \brief Element access
      * @param i row index of the element
      * @param j column index of the element
      * Returns a reference to the element located at (i,j)
      *
      * @precond i < num_rows(m) && j < num_cols(m)
      * @postcond i < num_rows(m) && j < num_cols(m)
      * @semantics No idea what to put here
      * @new_in{2.1}
      * We have introduced concept archetypes with automatic 
      * documentation derivation as well the ability to list new and 
      * changed things.
      * @complexity_worst{n^2} if the implementation is wrong
      * @complexity_worst{n} if only parts of the implementation is wrong
      * @complexity_average{1} if you implemented the routine nicely.
      **/
    value_type& operator()(size_type i, size_type j) { return value_type(); }

    /**
      * \brief Element access (constant)
      * @param i row index of the element
      * @param j column index of the element
      * @precond i < num_rows(m) && j < num_cols(m)
      * @new_in{2.3}
      * Returns a const reference to the element located at (i,j)
      * @requirement{i} should be bigger than 100000
      * @requirement{j} should be bigger than 100000. Otherwise 
      * you are just solving toy problems. 
      **/
    value_type const& operator()(size_type i, size_type j) const { return value_type(); }

    /**
      * \brief Assignement operator
      * Assigns the matrix to the argument
      * @return A reference to this.
      *
      * @postcond The matrix has the same dimensions as m and the same coefficients.
      * @invariant m remains unchanged.
      * @concepttitle{Assign}
      *
      **/
    matrix_archetype& operator = (matrix_archetype const& m) { return *this; }

    /**
      * \brief Plus-assignemnt
      * Adds the matrix m to the matrix
      *
      * @precond The matrices have the same dimensions
      * @postcond TODO
      *
      * @return A reference to this.
      * @concepttitle{Plus assign}
      */
    matrix_archetype& operator += (matrix_archetype const& m){ return *this; }

    // TODO complete this
};

/**
  * \brief Returns the number of rows

  * @invariant m remains unchanged
  * @concepttitle{Row count}
  * @complexity_worst{n^n} 
  * @complexity_average{1}
 **/
template <typename T>
typename matrix_archetype<T>::size_type num_rows(matrix_archetype<T> const& m) { return typename matrix_archetype<T>::size_type(0); }

/**
  * \brief Returns the number of columns
  * @invariant m remains unchanged
  * @concepttitle{Col. count}
 **/
template <typename T>
typename matrix_archetype<T>::size_type num_cols(matrix_archetype<T> const& m) { return typename matrix_archetype<T>::size_type(0); }

/**
  * \brief Addition
  * @return Sum of the matrices a and b
  */
template <typename T>
matrix_archetype<T> operator + (matrix_archetype<T> a, matrix_archetype<T> const& b) { return matrix_archetype<T>(); }


/**
  * \brief Row access
  * @invariant all other rows remain untouched
  * @return a pair of row iterators indicating the begin and end of row i
  */
template <typename T>
std::pair<typename matrix_archetype<T>::row_element_iterator,typename matrix_archetype<T>::row_element_iterator>
row(matrix_archetype<T>& m, typename matrix_archetype<T>::size_type i)
{ return typename matrix_archetype<T>::row_element_iterator(); }

template <typename T>
std::pair<typename matrix_archetype<T>::const_row_element_iterator,typename matrix_archetype<T>::const_row_element_iterator>
row(matrix_archetype<T> const& m, typename matrix_archetype<T>::size_type i)
{ return typename matrix_archetype<T>::const_row_element_iterator(); }

/**
  * \brief Column access
  * @invariant all other columns remain untouched
  * @return a pair of column iterators indicating the begin and end of the column j
  */
template <typename T>
std::pair<typename matrix_archetype<T>::col_element_iterator,typename matrix_archetype<T>::col_element_iterator>
col(matrix_archetype<T>& m, typename matrix_archetype<T>::size_type j)
{ return typename matrix_archetype<T>::col_element_iterator(); }

template <typename T>
std::pair<typename matrix_archetype<T>::const_col_element_iterator,typename matrix_archetype<T>::const_col_element_iterator>
col(matrix_archetype<T> const& m, typename matrix_archetype<T>::size_type j)
{ return typename matrix_archetype<T>::const_col_element_iterator(); }

} // end namespace concepts
} // end namespace numeric
} // end namespace alps

#endif // ALPS_NUMERIC_MATRIX_MATRIX_CONCEPT_ARCHETYPE_HPP
