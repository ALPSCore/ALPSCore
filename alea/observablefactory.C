/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

/* $Id: observableset.C 3765 2010-01-22 14:58:52Z troyer $ */

#include <alps/config.h>

#include "observablefactory.h"
#include "signedobservable.h"
#include "nobinning.h"
#include "detailedbinning.h"
#include "simpleobseval.h"
#include "histogrameval.h"

#include "simpleobseval.ipp"
#include "abstractsimpleobservable.ipp"
#include "simpleobservable.ipp"

namespace alps {

ObservableFactory::ObservableFactory()
{
  register_observable<IntObsevaluator>();
  register_observable<RealObsevaluator>();
  register_observable<IntObservable>();
  register_observable<RealObservable>();
  register_observable<AbstractSignedObservable<RealObsevaluator> >();
  register_observable<SignedObservable<RealObservable> >();
  register_observable<SignedObservable<SimpleRealObservable> >();
  register_observable<SignedObservable<RealTimeSeriesObservable> >();
  register_observable<IntTimeSeriesObservable>();
  register_observable<RealTimeSeriesObservable>();
  register_observable<SimpleRealObservable>();
  register_observable<SimpleIntObservable>();
  register_observable<RealVectorObsevaluator>();
  register_observable<RealVectorObservable>();
  register_observable<SignedObservable<RealVectorObservable> >();
  register_observable<IntVectorObservable>();
  register_observable<IntVectorObsevaluator>();
  register_observable<IntVectorTimeSeriesObservable>();
  register_observable<RealVectorTimeSeriesObservable>();
  register_observable<SimpleIntVectorObservable>();
  register_observable<SimpleRealVectorObservable>();
  register_observable<AbstractSignedObservable<RealVectorObsevaluator> >();
  register_observable<SignedObservable<SimpleRealVectorObservable> >();
  register_observable<SignedObservable<RealVectorTimeSeriesObservable> >();
  register_observable<IntHistogramObservable>();
  register_observable<RealHistogramObservable>();
  register_observable<IntHistogramObsevaluator>();
  register_observable<RealHistogramObsevaluator>();
/*
  register_observable<HistogramObservable<int32_t> >();
  register_observable<HistogramObservable<int32_t,double> >();
*/
}

} // namespace alps
