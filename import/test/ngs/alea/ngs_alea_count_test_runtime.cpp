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

BOOST_AUTO_TEST_CASE(test_count_in_modular_accum)
{
    alps::accumulator::accumulator<int> acci;
    
    for (int i = 0; i < 100; ++i)
        acci << i;
        
    BOOST_REQUIRE( alps::accumulator::count(acci) == 100);
    
    
    alps::accumulator::accumulator<double> accd;
    
    for (double i = 0; i < 1; i += .01)
        accd << i;
            
    BOOST_REQUIRE( alps::accumulator::count(accd) == 100);
}
