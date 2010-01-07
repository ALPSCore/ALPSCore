/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2004 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@itp.phys.ethz.ch>,
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

#ifndef ALPS_ALEA_RECORDABLEOBSERVABLE_H
#define ALPS_ALEA_RECORDABLEOBSERVABLE_H

#include <alps/config.h>
#include <boost/throw_exception.hpp>
#include <stdexcept>

namespace alps {

//=======================================================================
// RecordableObservable
//
// an observable that can store new measurements
//-----------------------------------------------------------------------

template <class T=double, class SIGN=double>
class RecordableObservable
{
public:
  typedef T value_type;
  typedef SIGN sign_type;

  /// just a default constructor  
  RecordableObservable() {}
  virtual ~RecordableObservable() {}

  /// add another measurement to the observable
  virtual void operator<<(const value_type& x) =0;
  /// add another measurement to the observable
  virtual void add(const value_type& x) { operator<<(x);}
  /// add an explcitly signed measurement to the observable
  virtual void add(const value_type& x, sign_type s) { 
    if (s==1)
      add(x);
    else
      boost::throw_exception(std::logic_error("Called add of unsigned dobservable with a sign that is not 1"));
  }
 
   };
}

#endif // ALPS_ALEA_SIMPLEOBSERVABLE_H
