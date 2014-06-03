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

#include <iostream>

BOOST_AUTO_TEST_CASE(test_stream_for_modular_accum)
{
    typedef alps::accumulator::accumulator<int, alps::accumulator::features<alps::accumulator::tag::mean> > accum;
    accum acc;
    
    acc << 1;
    acc << 2;
    std::cout << acc << std::endl;
    
    alps::accumulator::detail::accumulator_wrapper m(acc);

    std::cout << m << std::endl;   
}
