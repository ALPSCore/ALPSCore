/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_default_vectors.cpp

    @brief Tests vector parameters with default values
*/

#include <alps/params.hpp>
#include <gtest/gtest.h>

TEST(ParamsTestDefaultVectors, defvec) {
    alps::params p;
    typedef std::vector<int> intvec_t;
    p.define<intvec_t>("intvec", intvec_t({1,2,3}), "An int vector");
    intvec_t expected={1,2,3};
    intvec_t actual=p["intvec"];
    EXPECT_EQ(expected, actual);
}
