/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/ngs/config.hpp>

#ifndef ALPS_NGS_USE_NEW_ALEA
#error "This test only works with new alea library"
#endif

#define BOOST_TEST_MODULE alps::ngs::accumulator

#include <alps/ngs/accumulator/accumulator.hpp>

#ifndef ALPS_LINK_BOOST_TEST
#	include <boost/test/included/unit_test.hpp>
#else
#	include <boost/test/unit_test.hpp>
#endif

BOOST_AUTO_TEST_CASE(mean_feature) {

	alps::accumulator::accumulator_set measurements;
	measurements << alps::accumulator::RealObservable("obs1")
				 << alps::accumulator::RealObservable("obs2");

	for (int i = 1; i < 1000; ++i) {
		measurements["obs1"] << 1.;
		BOOST_REQUIRE(measurements["obs1"].mean<double>() == 1.);
		measurements["obs2"] << i;
		BOOST_REQUIRE(measurements["obs2"].mean<double>() == double(i + 1) / 2.);
	}

	alps::accumulator::result_set results(measurements);
	BOOST_REQUIRE(results["obs1"].mean<double>() == 1.);
	BOOST_REQUIRE(results["obs2"].mean<double>() == 500.);
}
