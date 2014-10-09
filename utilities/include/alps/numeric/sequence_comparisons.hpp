/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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

