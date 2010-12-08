/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_ALEA_ABSTRACTSIMPLEOBSERVABLE_IPP
#define ALPS_ALEA_ABSTRACTSIMPLEOBSERVABLE_IPP

#include <alps/alea/abstractsimpleobservable.h>
#include <alps/hdf5/valarray.hpp>

namespace alps {
#ifdef ALPS_HAVE_HDF5

template <class T>
void AbstractSimpleObservable<T>::serialize(hdf5::iarchive & ar) 
{
    Observable::serialize(ar);
    if (ar.is_data("labels"))
        ar >> make_pvp("labels", label_);
}
    
template <class T>
void AbstractSimpleObservable<T>::serialize(hdf5::oarchive & ar) const 
{
  Observable::serialize(ar);
  if (count() > 0) {
      if (label_.size())
          ar
              << make_pvp("labels", label_)
          ;
      ar
          << make_pvp("count", count())
          << make_pvp("mean/value", mean())
      ;
  }
  if (count() > 1) {
      ar
          << make_pvp("mean/error", error())
          << make_pvp("mean/error_convergence", converged_errors())
      ;
      if(has_variance())
          ar
              << make_pvp("variance/value", variance())
          ;
      if(has_tau())
          ar
              << make_pvp("tau/value", tau())
          ;
  }
}
#endif
}
#endif // ALPS_ALEA_ABSTRACTSIMPLEOBSERVABLE_IPP
