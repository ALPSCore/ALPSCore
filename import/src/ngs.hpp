/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                           Matthias Troyer <troyer@comp-phys.org>                *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef ALPS_NGS_HPP
#define ALPS_NGS_HPP

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/map.hpp>
#include <alps/hdf5/pair.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/pointer.hpp>
#include <alps/hdf5/complex.hpp>

#include <alps/ngs/api.hpp>
#include <alps/ngs/cast.hpp>
#include <alps/ngs/sleep.hpp>
#include <alps/ngs/signal.hpp>
#include <alps/ngs/random01.hpp>
#include <alps/ngs/boost_mpi.hpp>
#include <alps/ngs/short_print.hpp>
#include <alps/ngs/thread_exceptions.hpp>
#include <alps/ngs/observablewrappers.hpp> // TODO: remove!

#ifdef ALPS_NGS_USE_NEW_ALEA
	#include <alps/ngs/accumulator/accumulator.hpp>
#else
	namespace alps {
		namespace accumulator {

			typedef ::alps::ngs::SimpleRealObservable SimpleRealObservable;
			typedef ::alps::ngs::SimpleRealVectorObservable SimpleRealVectorObservable;

			typedef ::alps::ngs::RealObservable RealObservable;
			typedef ::alps::ngs::RealVectorObservable RealVectorObservable;

			typedef ::alps::ngs::SignedRealObservable SignedRealObservable;
			typedef ::alps::ngs::SignedRealVectorObservable SignedRealVectorObservable;

			typedef ::alps::ngs::SignedSimpleRealObservable SignedSimpleRealObservable;
			typedef ::alps::ngs::SignedSimpleRealVectorObservable SignedSimpleRealVectorObservable;
		}
	}
#endif

// #include <alps/mcbase.hpp>
// #include <alps/parseargs.hpp>
// #include <alps/stop_callback.hpp>
// #include <alps/progress_callback.hpp> // TODO: remove this file!

// TODO: remove these deprecated headers:
#include <alps/ngs/mcresult.hpp>
#include <alps/ngs/mcresults.hpp>
#include <alps/ngs/mcoptions.hpp>
#include <alps/ngs/mcobservable.hpp>
#include <alps/ngs/mcobservables.hpp> // TODO: rethink this!

#endif
