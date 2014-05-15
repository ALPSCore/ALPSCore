/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
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


/* $Id: nobinning.h 3520 2009-12-11 16:49:53Z gamperl $ */

#ifndef ALPS_VECTOR_VALARRAY_CONVERSION
#define ALPS_VECTOR_VALARRAY_CONVERSION

#include <alps/utility/data.hpp>
#include <vector>
#include <valarray>
#include <algorithm>


namespace alps {

  template <class T>
  struct vector2valarray_type {
    typedef T type;
  };

  template <class T, class A>
  struct vector2valarray_type<std::vector<T, A> > {
    typedef std::valarray<T> type;
  };


  template <class T>
  struct valarray2vector_type {
    typedef T type;
  };

  template <class T>
  struct valarray2vector_type<std::valarray<T> > {
    typedef std::vector<T> type;
  };


  namespace numeric {

    template <class T>
    T valarray2vector (const T& value) {return value;}

    template<class T>
    std::vector<T> valarray2vector(std::valarray<T> const & from)
    {
      std::vector<T> to;
      to.reserve(from.size());
      std::copy(data(from),data(from)+from.size(),std::back_inserter(to));
      return to;
    }



    template <class T>
    T vector2valarray (const T& value) {return value;}

    template<class T>
    std::valarray<T> vector2valarray(std::vector<T> const & from)
    {
      std::valarray<T> to(from.size());
      std::copy(from.begin(),from.end(),data(to));
      return to;
    }

    template<class T1, class T2>
    std::valarray<T2> vector2valarray(std::vector<T1> const & from)
    {
      std::valarray<T2> to(from.size());
      std::copy(from.begin(),from.end(),data(to));
      return to;
    }

  }
}

#endif
