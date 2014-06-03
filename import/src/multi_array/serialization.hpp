/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_MULTI_ARRAY_SERIALIZATION_HPP
#define ALPS_MULTI_ARRAY_SERIALIZATION_HPP

#include <alps/multi_array/multi_array.hpp>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/serialization.hpp>

namespace alps{

  template<typename Archive, typename T, std::size_t D, class Allocator> 
  inline void save(Archive & ar, const multi_array<T,D,Allocator> & t, const unsigned int file_version) 
  { 
    using boost::serialization::make_nvp;
    using boost::serialization::make_array;
    ar << make_nvp("dimensions", make_array(t.shape(), D)); 
    ar << make_nvp("data", make_array(t.data(), t.num_elements()));
  } 

  template<typename Archive, typename T, std::size_t D, class Allocator> 
  inline void load(Archive & ar, multi_array<T,D,Allocator> & t, const unsigned int file_version) 
  { 
    using boost::serialization::make_nvp;
    using boost::serialization::make_array;

    typedef typename multi_array<T,D,Allocator>::size_type size_type;

    boost::array<size_type, D> dimensions;
    ar >> make_nvp("dimensions", make_array(dimensions.c_array(), D));
    t.resize(dimensions); 
    ar >> make_nvp("data", make_array(t.data(), t.num_elements())); 
  } 

  template<typename Archive, typename T, std::size_t D, class Allocator> 
  inline void serialize(Archive & ar, multi_array<T,D,Allocator>& t, const unsigned int file_version) 
  { 
    using boost::serialization::split_free;
    
    split_free(ar, t, file_version); 
  }

}//namespace alps

#endif // ALPS_MULTI_ARRAY_SERIALIZATION_HPP
