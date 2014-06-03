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

#include "error_archetype.hpp"
  

BOOST_AUTO_TEST_CASE(test_fixed_size_bin_with_error_archetype)
{
    alps::accumulator::accumulator<error_archetype, alps::accumulator::features<alps::accumulator::tag::fixed_size_binning> > acci; //Default 128
    
    acci << 2;
    acci << 6;
    
    //~ BOOST_ERROR( fixed_size_binning(acci).bin_size() == 128);
    
    
    alps::accumulator::accumulator<error_archetype, alps::accumulator::features<alps::accumulator::tag::fixed_size_binning> > accd(alps::accumulator::bin_size = 10);

    accd << .2;
    accd << .6;
        
    //~ BOOST_REQUIRE( alps::accumulator::fixed_size_binning(accd).bin_size() == 10);
    
    alps::accumulator::fixed_size_binning(accd);
    accd.fixed_size_binning();
}
