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

BOOST_AUTO_TEST_CASE(test_error_with_error_archetype)
{
    typedef alps::accumulator::accumulator<error_archetype, alps::accumulator::features<alps::accumulator::tag::error> > accum;
    accum acc;
    
    for(int i = 0; i < 10; ++i)
        acc << error_archetype();
    
    error(acc);
    acc.error();
}
