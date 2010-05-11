/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2010 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef PARAPACK_MEASUREMENT_H
#define PARAPACK_MEASUREMENT_H

#include <alps/alea.h>

namespace alps {

ALPS_DECL void merge_clone(alps::ObservableSet& total, alps::ObservableSet const& clone,
  bool same_weight);

ALPS_DECL void merge_random_clone(alps::ObservableSet& total, alps::ObservableSet const& clone);

template<typename T>
void add_constant(Observable& obs, T const& val) {
  if (dynamic_cast<SimpleRealObservable*>(&obs) &&
      dynamic_cast<SimpleRealObservable*>(&obs)->count() < 2) obs << val;
}

} // end namespace alps

#endif // PARAPACK_MEASUREMENT_H
