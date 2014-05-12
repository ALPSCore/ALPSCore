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

template <class T,class BINNING> 
void SimpleObservable<T,BINNING>::save(hdf5::archive & ar) const 
{
  AbstractSimpleObservable<T>::save(ar);
  ar[""] << b_;
}

template <class T,class BINNING> 
void SimpleObservable<T,BINNING>::load(hdf5::archive & ar) 
{
  AbstractSimpleObservable<T>::load(ar);
  ar[""] >> b_;
}

template <class T,class BINNING> 
hdf5::archive & operator<<(hdf5::archive & ar,  SimpleObservable<T,BINNING> const& obs) 
{
  return ar["/simulation/results/" + obs.representation()] << obs;
}

template <class T,class BINNING> 
hdf5::archive & operator>>(hdf5::archive & ar,  SimpleObservable<T,BINNING>& obs) 
{
  return ar["/simulation/results/" + obs.representation()] >> obs;
}

} // end namespace alps

#endif // ALPS_ALEA_SIMPLEOBSERVABLE_IPP
