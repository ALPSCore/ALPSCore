/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2012 by Ilia Zintchenko <iliazin@gmail.com>                       *
 *                       Jan Gukelberger                                           *
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

#ifndef ALPS_MULTI_ARRAY_BASE_HPP
#define ALPS_MULTI_ARRAY_BASE_HPP

#include <boost/multi_array.hpp>

namespace alps{

  template <class T,std::size_t D>
  class multi_array : public boost::multi_array<T,D>
  {
    typedef boost::multi_array<T,D> base_type;

  public:

    multi_array(std::size_t N, std::size_t M, std::size_t K, std::size_t J) : base_type(boost::extents[N][M][K][J]) {}
    multi_array(std::size_t N, std::size_t M, std::size_t K) : base_type(boost::extents[N][M][K]) {}
    multi_array(std::size_t N, std::size_t M) : base_type(boost::extents[N][M]) {}
    multi_array(std::size_t N) : base_type(boost::extents[N]) {}

    multi_array(const boost::detail::multi_array::extent_gen<D>& ext) : base_type(ext) {}

    multi_array() : base_type() {}

    multi_array<T,D>& operator=(const multi_array<T,D>& a)
    {
      if(this != &a){
      	std::vector<std::size_t> ext(a.shape(),a.shape()+a.num_dimensions());
      	(*this).resize(ext);
      	base_type::operator=(a);
      }

      return *this;
    }

    multi_array<T,D>& operator+=(const multi_array<T,D>& a)
    {
      assert(std::equal(this->shape(),this->shape()+D,a.shape()));
      std::transform((*this).data(),(*this).data()+(*this).num_elements(),a.data(),(*this).data(),std::plus<T>());
      return *this;
    }

    multi_array<T,D>& operator-=(const multi_array<T,D>& a)
    {
      assert(std::equal(this->shape(),this->shape()+D,a.shape()));
      std::transform((*this).data(),(*this).data()+(*this).num_elements(),a.data(),(*this).data(),std::minus<T>());
      return *this;
    }

    multi_array<T,D>& operator*=(const multi_array<T,D>& a)
    {
      assert(std::equal(this->shape(),this->shape()+D,a.shape()));
      std::transform((*this).data(),(*this).data()+(*this).num_elements(),a.data(),(*this).data(),std::multiplies<T>());
      return *this;
    }

    multi_array<T,D>& operator*=(const T s)
    {
      std::transform((*this).data(),(*this).data()+(*this).num_elements(),(*this).data(),std::bind2nd(std::multiplies<T>(),s));
      return *this;
    }

    multi_array<T,D>& operator/=(const multi_array<T,D>& a)
    {
      assert(std::equal(this->shape(),this->shape()+D,a.shape()));
      std::transform((*this).data(),(*this).data()+(*this).num_elements(),a.data(),(*this).data(),std::divides<T>());
      return *this;
    }

    multi_array<T,D>& operator/=(const T s)
    {
      std::transform((*this).data(),(*this).data()+(*this).num_elements(),(*this).data(),std::bind2nd(std::divides<T>(),s));
      return *this;
    }
    
  };//class multi_array

}//namespace alps

#endif // ALPS_MULTI_ARRAY_BASE_HPP
