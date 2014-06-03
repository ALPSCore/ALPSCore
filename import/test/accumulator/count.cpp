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

BOOST_AUTO_TEST_CASE(count_feature) {

	alps::accumulator::accumulator_set measurements;
	measurements << alps::accumulator::RealObservable("scalar")
				 << alps::accumulator::RealVectorObservable("vector");

	for (int i = 1; i < 1001; ++i) {
		measurements["scalar"] << i;
		BOOST_REQUIRE(count(measurements["scalar"]) == i);
		measurements["vector"] << std::vector<double>(10, i);
		BOOST_REQUIRE(count(measurements["vector"]) == i);
	}

	alps::accumulator::result_set results(measurements);
	BOOST_REQUIRE(count(results["scalar"]) == 1000);
	BOOST_REQUIRE(count(results["vector"]) == 1000);
}
