/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */



#ifndef ALPS_NUMERIC_FOURIER_HPP
#define ALPS_NUMERIC_FOURIER_HPP

#include <alps/numeric/vector_functions.hpp>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>


namespace alps {
namespace numeric {

struct fourier_real
{
  template <class S, class T>
  static T
    evaluate
      (  std::vector<T>  const &  A    //  [a0, a1, a2,...]
      ,  std::vector<T>  const &  B    //  [b0, b1, b2,...]
      ,  T                const &  x    //  x
      ,  S                const & N    //  N
      )
    {
      T  _2PIx_over_N  =  ((M_2PI * x) / N);

      std::vector<T>  _2PInx_over_N;
      _2PInx_over_N.reserve(A.size());
      std::transform
        (  boost::counting_iterator<T>(0)
        ,  boost::counting_iterator<T>(A.size())
        ,  std::back_inserter(_2PInx_over_N)
        , boost::lambda::_1 * _2PIx_over_N
        );  

      std::vector<T>  COS = alps::numeric::cos(_2PInx_over_N);
      std::vector<T>  SIN = alps::numeric::sin(_2PInx_over_N);

      using alps::numeric::operator*;

      COS = COS * A;
      SIN = SIN * B;

      return ( std::accumulate(COS.begin(), COS.end(), 0.) +  std::accumulate(SIN.begin()+1, SIN.end(), 0.) );
    } // objective:  F(x) = a0 + a1 * cos(2\pix/N) + a2 * cos(4\pix/N) + ... + b1 * sin(2\pix/N) + b2 * sin(4\pix/N) + ...
};


} // ending namespace numeric
} // ending namespace alps


#endif

