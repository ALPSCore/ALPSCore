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


#ifndef ALPS_NUMERIC_ACCUMULATE_IF_HPP
#define ALPS_NUMERIC_ACCUMULATE_IF_HPP

#include <iostream>
#include <iterator>


namespace alps {
namespace numeric {


template <class InputIterator, class BinaryOperation, class BinaryPredicate, class T>
T
  accumulate_if
    (  InputIterator    first
    ,  InputIterator    last
    ,  T                init
    ,  BinaryOperation  binary_op
    ,  BinaryPredicate  pred
    )
  {
    while (first != last)
    {
      if (pred(*first))
        init = binary_op(init, *first);
      ++first;
    }
    return init;
  }


template <class InputIterator, class BinaryPredicate, class T>
T
  accumulate_if
    (  InputIterator    first
    ,  InputIterator    last
    ,  T                init
    ,  BinaryPredicate  pred
    )
  {
    return accumulate_if(first, last, init, std::plus<T>(), pred);
  } 


} // ending namespace numeric
} // ending namespace alps


#endif
