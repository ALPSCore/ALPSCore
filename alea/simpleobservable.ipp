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

#ifndef ALPS_ALEA_SIMPLEOBSERVABLE_IPP
#define ALPS_ALEA_SIMPLEOBSERVABLE_IPP

#include <alps/alea/simpleobservable.h>

namespace alps {

#ifdef ALPS_HAVE_HDF5

  template <class T,class BINNING> 
  void SimpleObservable<T,BINNING>::serialize(hdf5::iarchive & ar) 
  {
    AbstractSimpleObservable<T>::serialize(ar);
    ar >> make_pvp("", b_);
  }
  template <class T,class BINNING> 
  void SimpleObservable<T,BINNING>::serialize(hdf5::oarchive & ar) const 
  {
    AbstractSimpleObservable<T>::serialize(ar);
    ar << make_pvp("", b_);
  }

  template <class T,class BINNING> 
  hdf5::oarchive & operator<<(hdf5::oarchive & ar,  SimpleObservable<T,BINNING> const& obs) 
  {
    return ar << make_pvp("/simulation/results/"+obs.representation(), obs);
  }

  template <class T,class BINNING> 
  hdf5::iarchive & operator>>(hdf5::iarchive & ar,  SimpleObservable<T,BINNING>& obs) 
  {
    return ar >> make_pvp("/simulation/results/"+obs.representation(), obs);
  }
  
#endif

} // end namespace alps

#endif // ALPS_ALEA_SIMPLEOBSERVABLE_IPP
