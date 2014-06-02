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

BOOST_AUTO_TEST_CASE(test_log_bin_in_modular_accum)
{
    alps::accumulator::accumulator<int, alps::accumulator::features<alps::accumulator::tag::log_binning> > acci; //Default 128
    
    acci << 2;
    acci << 6;
    
    //~ BOOST_REQUIRE( alps::accumulator::max_num_binning(acci) == 128);
    
    
    alps::accumulator::accumulator<double, alps::accumulator::features<alps::accumulator::tag::log_binning> > accd;
    
    for(int i = 0; i < 96; ++i)
    {
        accd << 1.;
    }
    
    std::vector<double> vec;
    vec = log_binning(accd).bins();
    
    for(int i = 0; i < vec.size(); ++i)
    {
        std::cout << vec[i] << std::endl;
    }
        
    //~ BOOST_REQUIRE( alps::accumulator::max_num_binning(accd) == 10);
}
