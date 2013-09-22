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
#ifndef ALPS_NUMERIC_MATRIX_COLUMN_VIEW_HPP
#define ALPS_NUMERIC_MATRIX_COLUMN_VIEW_HPP

#include <cassert>
#include <alps/numeric/matrix/entity.hpp>
#include <alps/numeric/matrix/operators/multiply.hpp>
#include <alps/numeric/matrix/operators/multiply_scalar.hpp>
#include <alps/numeric/matrix/operators/plus_minus.hpp>
#include <alps/numeric/matrix/operators/op_assign.hpp>
#include <alps/numeric/matrix/operators/op_assign_vector.hpp>
#include <alps/numeric/matrix/detail/print_vector.hpp>
#include <alps/numeric/matrix/detail/column_view_adaptor.hpp>
#include <alps/numeric/matrix/vector.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_const.hpp>

namespace alps {
namespace numeric {

template <typename Matrix>
class column_view {
  public:
    typedef typename Matrix::value_type         value_type;       // The type T of the elements of the matrix
    typedef typename Matrix::reference          reference;        // Reference to value_type
    typedef typename Matrix::const_reference    const_reference;  // Const reference to value_type
    typedef typename Matrix::size_type          size_type;        // Unsigned integer type that represents the dimensions of the matrix
    typedef typename Matrix::difference_type    difference_type;  // Signed integer type to represent the distance of two elements in the memory
    typedef typename Matrix::const_col_element_iterator const_iterator;
    typedef typename boost::mpl::if_<
                  boost::is_const<Matrix>
                , const_iterator
                , typename Matrix::col_element_iterator
                >::type                                 iterator;

    explicit column_view(Matrix& m, size_type j)
    : col_(col(m,j))
    {
    }

    iterator begin() {
        return col_.first;
    }
    iterator end() {
        return col_.second;
    }
    const_iterator begin() const {
        return col_.first;
    }
    const_iterator end() const {
        return col_.second;
    }

    inline reference operator[](size_type i)
    {
        assert( i <  std::distance(col_.first,col_.second) );
        return *(col_.first + i);
    }

    inline const_reference operator[](size_type i) const
    {
        assert( i <  std::distance(col_.first,col_.second) );
        return *(col_.first + i);
    }

    inline reference operator()(std::size_t i)
    {
        return this->operator[](i);
    }

    inline const_reference operator()(std::size_t i) const
    {
        return this->operator[](i);
    }

    size_type size() const
    {
        using std::distance;
        return distance(col_.first,col_.second);
    }

    template <typename T2>
    column_view& operator += (T2 const& rhs)
    {
        plus_assign(*this, rhs, typename get_entity<column_view>::type(), typename get_entity<T2>::type());
        return *this;
    }

    template <typename T2>
    column_view& operator -= (T2 const& rhs)
    {
        minus_assign(*this, rhs, typename get_entity<column_view>::type(), typename get_entity<T2>::type());
        return *this;
    }

    template <typename T2>
    column_view& operator *= (T2 const& x)
    {
        multiplies_assign(*this, x, typename get_entity<column_view>::type(), typename get_entity<T2>::type());
        return *this;
    }

  private:
    std::pair<iterator,iterator> col_;
};

template <typename Matrix>
std::ostream& operator << (std::ostream& os, column_view<Matrix> const& cv)
{
    detail::print_vector(os,cv);
    return os;
}

template <typename Matrix>
struct entity<column_view<Matrix> >
{
    typedef tag::vector type;
};

// Specialization of the conj function
template <typename Matrix>
vector<typename Matrix::value_type> conj(column_view<Matrix> const& cv)
{
    vector<typename Matrix::value_type> v(cv);
    conj_inplace(v);
    return v;
}

template <typename Matrix, typename Vector>
struct plus_minus_return_type<column_view<Matrix>,Vector, tag::vector, tag::vector>
{
    typedef vector<typename detail::auto_deduce_plus_return_type<typename Matrix::value_type,typename Vector::value_type>::type> type;
};

template <typename Matrix, typename Vector>
struct plus_minus_return_type<Vector, column_view<Matrix>, tag::vector, tag::vector>
{
    typedef vector<typename detail::auto_deduce_plus_return_type<typename Matrix::value_type,typename Vector::value_type>::type> type;
};

template <typename Matrix1, typename Matrix2>
struct plus_minus_return_type<column_view<Matrix1>, column_view<Matrix2>, tag::vector, tag::vector>
{
    typedef vector<typename detail::auto_deduce_plus_return_type<typename Matrix1::value_type,typename Matrix2::value_type>::type> type;
};

template <typename Matrix1, typename Matrix2>
struct multiply_return_type<Matrix1, column_view<Matrix2>, tag::matrix, tag::vector>
{
    typedef vector<typename detail::auto_deduce_multiply_return_type<typename Matrix1::value_type,typename Matrix2::value_type>::type> type;
};

template <typename Matrix>
struct supports_blas<column_view<Matrix> > : supports_blas<Matrix> {};

} // end namespace numeric
} // end namespace alps

#endif // ALPS_NUMERIC_MATRIX_COLUMN_VIEW_HPP
