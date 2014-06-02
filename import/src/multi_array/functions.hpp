/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_MULTI_ARRAY_FUNCTIONS_HPP
#define ALPS_MULTI_ARRAY_FUNCTIONS_HPP

#include <alps/multi_array/multi_array.hpp>
#include <alps/numeric/special_functions.hpp>

namespace alps {

#define ALPS_IMPLEMENT_FUNCTION(FUN)                                                                                            \
    template <class T, std::size_t D, class Allocator> multi_array<T,D,Allocator> FUN (multi_array<T, D, Allocator> arg) {      \
        using std:: FUN;                                                                                                        \
        std::transform(arg.data(), arg.data() + arg.num_elements(), arg.data(), std::ptr_fun<T, T>(FUN));                       \
        return arg;                                                                                                             \
    }

ALPS_IMPLEMENT_FUNCTION(sin)
ALPS_IMPLEMENT_FUNCTION(cos)
ALPS_IMPLEMENT_FUNCTION(tan)
ALPS_IMPLEMENT_FUNCTION(sinh)
ALPS_IMPLEMENT_FUNCTION(cosh)
ALPS_IMPLEMENT_FUNCTION(tanh)
ALPS_IMPLEMENT_FUNCTION(asin)
ALPS_IMPLEMENT_FUNCTION(acos)
ALPS_IMPLEMENT_FUNCTION(atan)
ALPS_IMPLEMENT_FUNCTION(abs)
ALPS_IMPLEMENT_FUNCTION(sqrt)
ALPS_IMPLEMENT_FUNCTION(exp)
ALPS_IMPLEMENT_FUNCTION(log)
ALPS_IMPLEMENT_FUNCTION(fabs)

#undef ALPS_IMPLEMENT_FUNCTION

#define ALPS_IMPLEMENT_FUNCTION(FUN)                                                                                            \
    template <class T, std::size_t D, class Allocator> multi_array<T, D, Allocator> FUN (multi_array<T, D, Allocator> arg) {    \
        using alps::numeric:: FUN ;                                                                                             \
        std::transform(arg.data(), arg.data() + arg.num_elements(), arg.data(), std::ptr_fun<T, T>(FUN));                       \
        return arg;                                                                                                             \
    }

ALPS_IMPLEMENT_FUNCTION(sq)
ALPS_IMPLEMENT_FUNCTION(cb)
ALPS_IMPLEMENT_FUNCTION(cbrt)

#undef ALPS_IMPLEMENT_FUNCTION

    template <class T1, class T2, std::size_t D, class Allocator>
    multi_array<T1,D,Allocator> pow(multi_array<T1,D,Allocator> a, T2 s)
  {
    std::pointer_to_binary_function <T1,T2,T1> PowObject (std::ptr_fun<T1,T2,T1>(std::pow));
    std::transform(a.data(),a.data()+a.num_elements(),a.data(),std::bind2nd(PowObject,s));
    return a;
  }

  template <class T, std::size_t D, class Allocator>
  T sum(multi_array<T,D,Allocator>& a)
  {
    return std::accumulate(a.data(),a.data()+a.num_elements(),0.,std::plus<T>());
  }

}//namespace alps

#endif // ALPS_MULTI_ARRAY_FUNCTIONS_HPP

