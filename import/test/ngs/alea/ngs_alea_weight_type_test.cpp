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

#include "hist_archetype.hpp"

#include <utility>


BOOST_AUTO_TEST_CASE(test_hist_with_weight_archetype)
{
    alps::accumulator::histogram_old<int, hist_archetype> acc(1, 6, 6);
    
    for(int i = 0; i < 10; ++i)
        acc << i;
    acc[1] += hist_archetype();
    acc[1] ++;
    ++acc[1];
    acc << std::pair<int, hist_archetype>(1,hist_archetype());
    std::cout << acc[1] << std::endl;
    std::cout << acc << std::endl;
    
    acc.count();    
    acc.mean();
}
