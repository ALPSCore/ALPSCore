/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
