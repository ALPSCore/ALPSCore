/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_MULTI_ARRAY_IO_HPP
#define ALPS_MULTI_ARRAY_IO_HPP

#include <alps/multi_array/multi_array.hpp>

namespace alps{

  template <class T, std::size_t D, class Allocator>
  std::ostream& operator<<(std::ostream& out, const multi_array<T,D,Allocator>& a)
  {
    boost::array<typename boost::multi_array<T,D,Allocator>::index, D> index;
    T* rE = const_cast< T* >(a.data());

    for(std::size_t dir = 0; dir < D; dir++) out << "{";
      
    for(int i = 0; i < a.num_elements(); i++)
      {
  	for(std::size_t dir = 0; dir < D; dir++ )
  	  index[dir] = (rE - a.origin()) / a.strides()[dir] % a.shape()[dir] +  a.index_bases()[dir];

	if(index[D-1] == a.shape()[D-1]-1){

	  int M = 1;
	  for(std::size_t dir = D-1; dir > 0; --dir){
	    if(index[dir-1] == a.shape()[dir-1]-1) M++;
	    else break;
	  }

	  out << a(index);

	  for(int m = 0; m < M; ++m)
	    out << "}";
	
	  if(M < D){
	    
	    if(M > 1 || M == D - 1)
	      out << ",\n";
	    else
	      out << ", ";  

	    for(int m = 0; m < M; ++m)
	      out << "{";  

	  }

	}
	else 
	  out << a(index) << ", ";

  	++rE;
      }

    out << ";";

    return out;
  }

}//namespace alps

#endif // ALPS_MULTI_ARRAY_IO_HPP
