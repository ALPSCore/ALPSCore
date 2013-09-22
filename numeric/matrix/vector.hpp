/*****************************************************************************
 *
 * ALPS DMFT Project - BLAS Compatibility headers
 *  Square Matrix Class
 *
 * Copyright (C) 2005 - 2012 by 
 *                              Emanuel Gull <gull@phys.columbia.edu>
 *                              Brigitte Surer <surerb@phys.ethz.ch>
 *                              Andreas Hehn <hehn@phys.ethz.ch>
 *
 *
 * This software is part of the ALPS Applications, published under the ALPS
 * Application License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 * 
 * You should have received a copy of the ALPS Application License along with
 * the ALPS Applications; see the file LICENSE.txt. If not, the license is also
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

#ifndef ALPS_VECTOR_HPP
#define ALPS_VECTOR_HPP

#include <alps/numeric/matrix/detail/vector_adaptor.hpp>
#include <alps/numeric/matrix/matrix_traits.hpp>

#include <alps/numeric/matrix/detail/blasmacros.hpp>
#include <boost/numeric/bindings/blas/level1/dot.hpp>
#include <ostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <alps/numeric/matrix/detail/print_vector.hpp>
#include <alps/numeric/scalar_product.hpp>


#include <alps/numeric/matrix/entity.hpp>
#include <alps/numeric/matrix/operators/op_assign.hpp>
#include <alps/numeric/matrix/operators/op_assign_vector.hpp>
#include <alps/numeric/matrix/operators/plus_minus.hpp>
#include <alps/numeric/matrix/detail/auto_deduce_plus_return_type.hpp>
#include <alps/numeric/matrix/exchange_value_type.hpp>

namespace alps {
namespace numeric {

  template<typename T, typename MemoryBlock = std::vector<T> >
  class vector : public MemoryBlock
  {
    public:
      explicit vector(std::size_t size=0, T const& initial_value = T())
      : MemoryBlock(size, initial_value)
      {
      }

      vector(vector const& v)
      : MemoryBlock(v)
      {
      }

      template <class InputIterator>
      vector (InputIterator first, InputIterator last)
      : MemoryBlock( first, last )
      {
      }

      template <typename Vector>
      explicit vector(Vector const& v, typename boost::enable_if<boost::is_same<typename get_entity<Vector>::type, tag::vector>, void>::type* = 0)
      : MemoryBlock(v.begin(), v.end())
      {
      }

      friend void swap(vector& x,vector& y)
      {
          std::swap(x, y);
      }

      inline T &operator()(const std::size_t i)
      {
          assert((i < this->size()));
          return this->operator[](i);
      }

      inline const T &operator()(std::size_t i) const
      {
          assert((i < this->size()));
          return this->operator[](i);
      }

      template <typename T2>
      vector& operator += (T2 const& rhs)
      {
          plus_assign(*this, rhs, typename get_entity<vector>::type(), typename get_entity<T2>::type());
          return *this;
      }

      template <typename T2>
      vector& operator-=(T2 const& rhs)
      {
          minus_assign(*this, rhs, typename get_entity<vector>::type(), typename get_entity<T2>::type());
          return *this;
      }

      template <typename T2>
      vector& operator *= (T2 const& x)
      {
          multiplies_assign(*this, x, typename get_entity<vector>::type(), typename get_entity<T2>::type());
          return *this;
      }
  };

    template<typename T, typename MemoryBlock>
    void insert(vector<T,MemoryBlock>& v, T value, std::size_t i)
    {
        assert((i <= v.size()));
        v.insert(v.begin()+i,value);
    }

    template <typename T, typename MemoryBlock>
    inline vector<T,MemoryBlock> exp(T c, vector<T,MemoryBlock> v)
    {
        vector<T,MemoryBlock> result(v.size());
        v*=c;
        std::transform(v.begin(), v.end(), result.begin(), static_cast<T(*)(T)> (&std::exp));
        return result;
    }

    template <typename MemoryBlock>
    inline vector<double,MemoryBlock> exp(double c, vector<double,MemoryBlock> v)
    {
        fortran_int_t s=v.size();
        vector<double,MemoryBlock> result(s);
        v*=c;
#ifdef VECLIB
        vecLib::vvexp(&result[0], &v[0], &s);
#else
#ifdef ACML
        acml::vrda_exp(s, &v[0], &result[0]);
#else
#ifdef MKL
        mkl::vdExp(s,  &v[0], &result[0]);
#else
        using std::exp;
        std::transform(v.begin(), v.end(), result.begin(), static_cast<double(*)(double)> (&exp));
#endif
#endif
#endif
        return result;
    }

  template <typename T, typename MemoryBlock>
  inline std::ostream &operator<<(std::ostream &os, const vector<T,MemoryBlock> &v)
  {
    detail::print_vector(os,v);
    return os;
  }


template <typename T, typename MemoryBlock>
struct entity<vector<T,MemoryBlock> >
{
    typedef tag::vector type;
};

template <typename T1, typename MemoryBlock1, typename T2, typename MemoryBlock2>
struct plus_minus_return_type<vector<T1,MemoryBlock1>, vector<T2,MemoryBlock2>, tag::vector, tag::vector>
{
    private:
        typedef typename detail::auto_deduce_plus_return_type<T1,T2>::type value_type;
        typedef typename boost::mpl::if_<typename detail::auto_deduce_plus_return_type<T1,T2>::select_first,MemoryBlock1,MemoryBlock2>::type memory_block_type;
    public:
        typedef vector<value_type, memory_block_type> type;
};

template <typename T, typename MemoryBlock, typename T2>
struct exchange_value_type<vector<T,MemoryBlock>,T2>
{
    typedef vector<T2> type;
};

#define ALPS_VECTOR_BLAS_TRAITS(T) \
template <typename MemoryBlock> \
struct supports_blas<vector<T,MemoryBlock> > : boost::mpl::true_ {};

ALPS_IMPLEMENT_FOR_ALL_BLAS_TYPES(ALPS_VECTOR_BLAS_TRAITS)

#undef ALPS_VECTOR_BLAS_TRAITS

   } //namespace numeric 
} //namespace alps

#endif //ALPS_VECTOR_HPP
