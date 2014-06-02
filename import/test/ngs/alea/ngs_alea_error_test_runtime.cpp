/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#define BOOST_TEST_MODULE alps::ngs::accumulator

#include <alps/ngs.hpp>

#ifndef ALPS_LINK_BOOST_TEST
#include <boost/test/included/unit_test.hpp>
#else
#include <boost/test/unit_test.hpp>
#endif
#include <boost/test/floating_point_comparison.hpp>

BOOST_AUTO_TEST_CASE(test_error_in_modular_accum)
{
    alps::accumulator::accumulator<int, alps::accumulator::features<alps::accumulator::tag::error> > acci;
    
    acci << 2;
    acci << 6;
    
    BOOST_REQUIRE( error(acci) == 2.);
    
    
    alps::accumulator::accumulator<double, alps::accumulator::features<alps::accumulator::tag::error> > accd;
    
    
    accd << .2;
    accd << .6;
    
    BOOST_REQUIRE_CLOSE(alps::accumulator::error(accd), 0.2, 0.01);
}
