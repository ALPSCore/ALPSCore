/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
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

    // left_raw_res_type& left_raw_res=left.extract<left_raw_res_type>();
    // right_raw_res_type& right_raw_res=right.extract<right_raw_res_type>();

    // right_raw_res_type& left_raw_res1=dynamic_cast<right_raw_res_type&>(left_raw_res);
    // right_raw_res_type& left_raw_res1=left.extract<right_raw_res_type>();
    
    // result_wrapper left1(left_raw_res1);
    result_wrapper left1=cast<left_raw_res_type,right_raw_res_type>(left);
    
    const result_wrapper r=left1+right;
    double xmean=r.mean<double>();
    EXPECT_EQ(2.,xmean);
}
