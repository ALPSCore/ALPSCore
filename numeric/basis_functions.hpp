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



#ifndef ALPS_NUMERIC_BASIS_FUNCTIONS_HPP
#define ALPS_NUMERIC_BASIS_FUNCTIONS_HPP

#include <alps/numeric/vector_functions.hpp>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>


namespace alps {
namespace numeric {


struct polynomial
{
  template <class T>
  static T
    evaluate
      ( std::vector<T>	const &	C
      , T								const &	x
      )
    {
      std::vector<T> X;
      X.reserve(C.size());
      X.push_back(1);
      for (std::size_t index=1; index < C.size(); ++index)
      {
        X.push_back(x * X[index-1]);
      }

      return std::inner_product(C.begin(), C.end(), X.begin(),	0.);
    } // objective: P(x) = C0 + C1 * x + C2 * x^2 + ... 


	template <class T>
	static T
		evaluate_derivative
      ( std::vector<T>	const &	C
      , T								const &	x
      )
    {
      std::vector<T> 	X;
      X.reserve(C.size());
      X.push_back(1);
      for (std::size_t index=1; index < C.size()-1; ++index)
      {
        X.push_back(x * X[index-1]);
      }
			
			std::vector<T>	C_prime;
			C_prime.reserve(C.size()-1);
			std::transform(boost::counting_iterator<unsigned int>(1), boost::counting_iterator<unsigned int>(C.size()), C.begin()+1, std::back_inserter(C_prime), boost::lambda::_1 * boost::lambda::_2);

			return std::inner_product(C_prime.begin(), C_prime.end(), X.begin(), 0.);
		}	
};


struct fourier_real
{
	template <class T>
	static T
		cos
		(	T & x
		)
	{
		return std::cos(x);
	}

	template <class T>
	static T
		sin
		(	T & x
		)
	{
		return std::sin(x);
	}


	template <class S, class T>
	static T
		evaluate
			(	std::vector<T>	const &	A		//	[a0, a1, a2,...]
			,	std::vector<T>	const &	B		//	[b0, b1, b2,...]
			,	T								const &	x		//	x
			,	S								const & N		//	N
			)
		{
			T	_2PIx_over_N	=	((M_2PI * x) / N);

			std::vector<T>	_2PInx_over_N;
			_2PInx_over_N.reserve(A.size());
			std::transform
				(	boost::counting_iterator<T>(0)
				,	boost::counting_iterator<T>(A.size())
				,	std::back_inserter(_2PInx_over_N)
				, boost::lambda::_1 * _2PIx_over_N
				);	

			std::vector<T>	C = alps::numeric::cos(_2PInx_over_N);
			std::vector<T>	S = alps::numeric::sin(_2PInx_over_N);

			using alps::numeric::operator*;

			C = C * A;
			S = S * B;

			return ( std::accumulate(C.begin(), C.end(), 0.) +  std::accumulate(S.begin()+1, S.end(), 0.) );
		} // objective:	F(x) = a0 + a1 * cos(2\pix/N) + a2 * cos(4\pix/N) + ... + b1 * sin(2\pix/N) + b2 * sin(4\pix/N) + ...
};


} // ending namespace numeric
} // ending namespace alps


#endif

