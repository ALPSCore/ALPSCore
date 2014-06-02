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

BOOST_AUTO_TEST_CASE(test_mean_in_modular_accum)
{
    alps::accumulator::accumulator<int, alps::accumulator::features<alps::accumulator::tag::mean> > acci;
    
    for(int i = 0; i < 101; ++i)
        acci << i;
        
    BOOST_REQUIRE( alps::accumulator::mean(acci) == 50);
    
    
    alps::accumulator::accumulator<double, alps::accumulator::features<alps::accumulator::tag::mean> > accd;
    
    for(double i = 0; i < 1.01; i += .01)
        accd << i;
        
    BOOST_REQUIRE( alps::accumulator::mean(accd) > .49999999999);
    BOOST_REQUIRE( alps::accumulator::mean(accd) < .50000000001);
}
