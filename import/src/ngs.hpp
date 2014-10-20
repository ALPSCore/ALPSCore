/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
	#include <alps/ngs/accumulators/accumulator.hpp>
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
