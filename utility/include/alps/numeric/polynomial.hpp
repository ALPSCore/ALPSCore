/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */



#ifndef ALPS_NUMERIC_POLYNOMIAL_HPP
#define ALPS_NUMERIC_POLYNOMIAL_HPP

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
      ( std::vector<T>  const &  C
      , T                const &  x
      )
    {
      std::vector<T> X;
      X.reserve(C.size());
      X.push_back(1);
      for (std::size_t index=1; index < C.size(); ++index)
      {
        X.push_back(x * X[index-1]);
      }

      return std::inner_product(C.begin(), C.end(), X.begin(),  0.);
    } // objective: P(x) = C0 + C1 * x + C2 * x^2 + ... 


  template <class T>
  static T
    evaluate_derivative
      ( std::vector<T>  const &  C
      , T                const &  x
      )
    {
      std::vector<T>   X;
      X.reserve(C.size());
      X.push_back(1);
      for (std::size_t index=1; index < C.size()-1; ++index)
      {
        X.push_back(x * X[index-1]);
      }
      
      std::vector<T>  C_prime;
      C_prime.reserve(C.size()-1);
      std::transform(boost::counting_iterator<unsigned int>(1), boost::counting_iterator<unsigned int>(C.size()), C.begin()+1, std::back_inserter(C_prime), boost::lambda::_1 * boost::lambda::_2);

      return std::inner_product(C_prime.begin(), C_prime.end(), X.begin(), 0.);
    }  
};


} // ending namespace numeric
} // ending namespace alps


#endif

