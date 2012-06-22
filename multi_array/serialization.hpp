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

#ifndef ALPS_MULTI_ARRAY_SERIALIZATION_HPP
#define ALPS_MULTI_ARRAY_SERIALIZATION_HPP

#include <boost/mpi.hpp>
#include <alps/multi_array/multi_array.hpp>

namespace alps{

  template<typename Archive, typename T, std::size_t D> 
  inline void save(Archive & ar, const multi_array<T, D> & t, const unsigned int file_version) 
  { 
    ar << boost::serialization::make_nvp("dimensions", boost::serialization::make_array(t.shape(), D)); 
    ar << boost::serialization::make_nvp("data", boost::serialization::make_array(t.data(), t.num_elements())); 
  } 

  template<typename Archive, typename T, std::size_t D> 
  inline void load(Archive & ar, multi_array<T, D> & t, const unsigned int file_version) 
  { 
    typedef typename multi_array<T, D>::size_type size_type; 

    boost::array<size_type, D> dimensions; 
    ar >> boost::serialization::make_nvp("dimensions", boost::serialization::make_array(dimensions.c_array(), D)); 
    t.resize(dimensions); 
    ar >> boost::serialization::make_nvp("data", boost::serialization::make_array(t.data(), t.num_elements())); 
  } 

  template<typename Archive, typename T, std::size_t D> 
  inline void serialize(Archive & ar, multi_array<T, D>& t, const unsigned int file_version) 
  { 
    boost::serialization::split_free(ar, t, file_version); 
  }


}//namespace alps

#endif // ALPS_MULTI_ARRAY_SERIALIZATION_HPP
