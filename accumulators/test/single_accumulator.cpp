/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file single_accumulator.cpp
    Test access of accumulator and result that is not a part of a set
*/


#include "gtest/gtest.h"
#include "alps/accumulators.hpp"


/// Google Test Fixture: A is a named accumulator type
template <typename A>
class SingleAccumulatorTest : public ::testing::Test {
  public:
    typedef A named_acc_type; ///< For example, `FullBinningAccumulator<double>`
    typedef typename named_acc_type::accumulator_type::value_type value_type; /// For example, `double`

    /// Way to access a single accumulator in a simple manner
    static void Simple()
    {
        named_acc_type named_acc("accname");
        named_acc << 1.0;
        EXPECT_EQ(1u, named_acc.result()->count());
        EXPECT_EQ(1.0,named_acc.result()->template mean<value_type>());
    }

    /// Way to access a single accumulator using the "old" features
    static void Elaborate()
    {
        using namespace alps::accumulators;
        named_acc_type named_acc("accname");
        std::shared_ptr<accumulator_wrapper> awptr=named_acc.wrapper;
        accumulator_wrapper& aw=*awptr;
        aw << 1.0;

        std::shared_ptr<result_wrapper> resptr=awptr->result();
        result_wrapper& res=*resptr;
        // This is wrong:
        // result_wrapper& res=*(awptr->result());
        int n=res.count();
        EXPECT_EQ(1, n);
        EXPECT_EQ(1.0, res.mean<value_type>());
    }

    /// Way to access a single accumulator via accumulator_set and result_set
    static void Conventional()
    {
        named_acc_type named_acc("accname");
        alps::accumulators::accumulator_set aset;
        aset << named_acc;
        alps::accumulators::accumulator_wrapper& aw=aset["accname"];
        aw << 1.0;

        alps::accumulators::result_set rset(aset);
        alps::accumulators::result_wrapper& res=rset["accname"];
        unsigned int n=res.count();
        EXPECT_EQ(1u, n);
        EXPECT_EQ(1.0, res.mean<value_type>());
    }

    /// Vector of named accumulators
    static void VectorOfAcc()
    {
        std::vector<named_acc_type> acc_vec(2,named_acc_type("")); // accumulators are copied here to fill the vector

        acc_vec[0] << 1.;

        EXPECT_EQ(1.0, acc_vec[0].result()->template mean<value_type>());
        EXPECT_EQ(1u, acc_vec[0].result()->count());
        EXPECT_EQ(0u, acc_vec[1].result()->count()); //verify that the wrapped accumulator is not shared
    }
};

typedef ::testing::Types<
    alps::accumulators::MeanAccumulator<double>,
    alps::accumulators::NoBinningAccumulator<double>,
    alps::accumulators::LogBinningAccumulator<double>,
    alps::accumulators::FullBinningAccumulator<double>
    > named_acc_test_types;

TYPED_TEST_CASE(SingleAccumulatorTest, named_acc_test_types);

TYPED_TEST(SingleAccumulatorTest, Conventional) { TestFixture::Conventional(); }
TYPED_TEST(SingleAccumulatorTest, Elaborate) { TestFixture::Elaborate(); }
TYPED_TEST(SingleAccumulatorTest, Simple) { TestFixture::Simple(); }
TYPED_TEST(SingleAccumulatorTest, VectorOfAcc) { TestFixture::VectorOfAcc(); }
