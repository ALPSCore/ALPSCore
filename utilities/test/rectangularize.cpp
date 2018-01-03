/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "alps/numeric/rectangularize.hpp"
#include <gtest/gtest.h>

TEST(Rectangularize,Rectangularize)
{
    typedef std::vector<double> vec_type;
    typedef std::vector<vec_type> mtx_type;

    vec_type row1(3, 1.5);
    vec_type row2(2, 2.25);
    mtx_type mtx(3);

    mtx[0]=row1;
    mtx[1]=row2;

    alps::numeric::rectangularize(mtx);

    mtx_type expected_mtx;
    expected_mtx.push_back(row1);
    expected_mtx.push_back(row2); expected_mtx[1].push_back(0);
    expected_mtx.push_back(vec_type(3, 0.0));

    EXPECT_EQ(expected_mtx, mtx);
}

    
