/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file binop_mixed_faildemo.cpp
    Test simple binary operations between results from different accumulators.
    This is a simplest example demonstrating the issue.
*/
#include <alps/accumulators.hpp>

#include <gtest/gtest.h>

TEST(AccumulatorMixedBinaryTest,add)
{
    using namespace alps::accumulators;

    typedef NoBinningAccumulator<double> left_acc_type;
    typedef MeanAccumulator<double> right_acc_type;

    typedef left_acc_type::result_type left_raw_res_type;
    typedef right_acc_type::result_type right_raw_res_type;

    accumulator_set aset;
    aset << left_acc_type("left")
         << right_acc_type("right");
    aset["left"] << 1.;
    aset["left"] << 1.;

    aset["right"] << 1.;
    aset["right"] << 1.;

    result_set rset(aset);
    result_wrapper& left=rset["left"];
    result_wrapper& right=rset["right"];

    result_wrapper left1=alps::accumulators::cast_raw<left_raw_res_type,right_raw_res_type>(left);
    // Does not compile, as expected:
    // result_wrapper right1=alps::accumulators::cast_raw<right_raw_res_type,left_raw_res_type>(right);

    // Alternatively:
    result_wrapper left2=alps::accumulators::cast<NoBinningAccumulator,MeanAccumulator>(left);

    const result_wrapper r1=left1+right;
    const result_wrapper r2=left2+right;
    double xmean1=r1.mean<double>();
    double xmean2=r2.mean<double>();
    EXPECT_EQ(2.,xmean1);
    EXPECT_EQ(2.,xmean2);
}
