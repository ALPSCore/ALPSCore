/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file assign_acc.cpp
    Test named accumulator assignment operation
**/

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"

namespace aa=alps::accumulators;

template <typename NAMED_ACC>
struct AccumulatorTest : public ::testing::Test  {
    typedef NAMED_ACC named_acc_type;
    typedef typename aa::value_type<typename named_acc_type::accumulator_type>::type value_type;

    static void assign_named() {
        named_acc_type rhs("rhs");
        named_acc_type lhs("lhs");
        lhs=rhs;
        aa::accumulator_set aset;
        aset << lhs;

        EXPECT_ANY_THROW(aa::accumulator_wrapper acc=aset["lhs"]; acc << 1 << 2);
        EXPECT_NO_THROW(aa::accumulator_wrapper acc=aset["rhs"]; acc << 1 << 2);
    }

    static void self_assign_named() {
        named_acc_type rhs("rhs");
        rhs=rhs;
        aa::accumulator_set aset;
        aset << rhs;

        EXPECT_NO_THROW(aa::accumulator_wrapper acc=aset["rhs"]; acc << 1 << 2);
    }
};

using AccTypes=::testing::Types<
    aa::MeanAccumulator<double>,
    aa::NoBinningAccumulator<double>,
    aa::LogBinningAccumulator<double>,
    aa::FullBinningAccumulator<double>
    >;

TYPED_TEST_CASE(AccumulatorTest, AccTypes);

TYPED_TEST(AccumulatorTest, AssignNamed) { this->assign_named(); }
TYPED_TEST(AccumulatorTest, SelfAssignNamed) { this->self_assign_named(); }
