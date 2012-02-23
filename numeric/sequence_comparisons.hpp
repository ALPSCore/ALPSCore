/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* Copyright (C) 2011-2012 by Lukas Gamper <gamperl@gmail.com>,
*                            Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Maximilian Poprawe <poprawem@ethz.ch>
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

#ifndef ALPS_UTILITY_SEQUENCE_COMPARISONS_HPP
#define ALPS_UTILITY_SEQUENCE_COMPARISONS_HPP

#include <alps/type_traits/is_sequence.hpp>
#include <alps/type_traits/element_type.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/bind.hpp>

#include <algorithm>

namespace alps {
namespace numeric {

  template <class X, class BinaryPredicate>                                                                                                                     
  inline typename boost::disable_if< is_sequence<X>, bool>::type                                                                         
        at_least_one (const X& value1, const X& value2, const BinaryPredicate& pred) {                                                                               
    return pred(value1, value2);                                                                                                  
  }                                                                                                                                      
                                                                                                                                         
  template <class X, class BinaryPredicate>                                                                                                                   
  inline typename boost::enable_if< is_sequence<X>, bool>::type                                                                          
        at_least_one (const X& sequence, const typename element_type<X>::type & value, const BinaryPredicate& pred) {                                                
    return sequence.end() != std::find_if(sequence.begin(), sequence.end(), boost::bind<bool>(pred, boost::lambda::_1, value) );                      
  }                                                                                                                                      
                                                                                                                                         
  template <class X, class BinaryPredicate>                                                                                                                   
  inline typename boost::enable_if< is_sequence<X>, bool>::type                                                                          
        at_least_one (const typename element_type<X>::type & value, const X& sequence, const BinaryPredicate& pred) {                                                
    return sequence.end() != std::find_if(sequence.begin(), sequence.end(), boost::bind<bool>(pred, value, boost::lambda::_1 ) );                      
  }                                                                                                                                      
                                                                                                                                         
  template <class X, class BinaryPredicate>                                                                                                                    
  inline typename boost::enable_if< is_sequence<X>, bool>::type                                                                          
        at_least_one (const X& sequence1, const X& sequence2, const BinaryPredicate& pred) {                                                                         
    return !(std::equal(sequence1.begin(), sequence1.end(), sequence2.begin(), !boost::bind<bool>(pred, boost::lambda::_1, boost::lambda::_2)));     
  }


} // ending namespace numeric
} // ending namespace alps

#endif

