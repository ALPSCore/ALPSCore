/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"
// This tets was written to expose a bug occuring with number of measurements is between 64 and 127

// using Google Test Fixture
class AccumulatorTest : public ::testing::Test {
    public:
    template <typename A>
    void tester(const unsigned int nsamples)
    {
        typedef typename alps::accumulators::value_type<typename A::accumulator_type>::type value_type;
        alps::accumulators::accumulator_set measurements;
        measurements<<A("one_half");
        for(unsigned int count=0; count<nsamples; ++count){
            measurements["one_half"]<<0.5;
        }

        std::shared_ptr<alps::accumulators::result_wrapper> res=measurements["one_half"].result();
        value_type xmean=res->mean<value_type>();

        EXPECT_NEAR(value_type(0.5), xmean, 1E-8);
        EXPECT_EQ(nsamples, res->count());
    }
};


#define MAKE_TEST(atype,dtype,num) \
    TEST_F(AccumulatorTest, Result ## atype ## X ## dtype ## num) { tester< alps::accumulators::atype<dtype> >(num); }

MAKE_TEST(MeanAccumulator,double, 128)
MAKE_TEST(NoBinningAccumulator,double, 128)
MAKE_TEST(LogBinningAccumulator,double, 128)
MAKE_TEST(FullBinningAccumulator,double, 128)

MAKE_TEST(MeanAccumulator,double, 1)
MAKE_TEST(NoBinningAccumulator,double, 1)
MAKE_TEST(LogBinningAccumulator,double, 1)
MAKE_TEST(FullBinningAccumulator,double, 1)

MAKE_TEST(MeanAccumulator,double, 63)
MAKE_TEST(NoBinningAccumulator,double, 63)
MAKE_TEST(LogBinningAccumulator,double, 63)
MAKE_TEST(FullBinningAccumulator,double, 63)

MAKE_TEST(MeanAccumulator,double, 127)
MAKE_TEST(NoBinningAccumulator,double, 127)
MAKE_TEST(LogBinningAccumulator,double, 127)
MAKE_TEST(FullBinningAccumulator,double, 127)
