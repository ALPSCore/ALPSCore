/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010 by Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
*                       Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

