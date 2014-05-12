/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@comp-phys.org>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@comp-phys.org>,
*                            Synge Todo <wistaria@comp-phys.org>
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

/* $Id$ */

#ifndef ALPS_LAMBDA_HPP
#define ALPS_LAMBDA_HPP

#include <boost/lambda/lambda.hpp>
#include <valarray>

namespace boost { 
namespace lambda {
  
template<class Act, class T> 
struct plain_return_type_2<arithmetic_action<Act>, std::valarray<T>, std::valarray<T> > {
  typedef std::valarray<T> type;
};

template<class Act, class T, class U> 
struct plain_return_type_2<arithmetic_action<Act>, std::valarray<T>, U> {
  typedef std::valarray<T> type;
};

template<class Act, class T, class U> 
struct plain_return_type_2<arithmetic_action<Act>, U, std::valarray<T> > {
  typedef std::valarray<T> type;
};

}
}

#endif // ALPS_LAMBDA_HPP
