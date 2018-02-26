/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/accumulators.hpp>
#include "gtest/gtest.h"

#include "accumulator_generator.hpp"

namespace aa=alps::accumulators;
namespace aat=alps::accumulators::testing;

template <typename A>
class NegativeErrorTest : public ::testing::Test {
  public:
    typedef aat::AccResultGenerator< A, 100000, aat::CorrelatedData<10> > acc_gen_type;
    acc_gen_type acc_gen;

    void negative_error_test() {
        const aa::result_wrapper& res=acc_gen.result();
        aa::result_wrapper neg_res = -res;
        std::cout << "Original result: " << res << std::endl
                  << "Negated result: " << neg_res << std::endl;
            
        ASSERT_TRUE(neg_res.error<double>() > 0) << "Error bar is negative";
        EXPECT_NEAR(acc_gen.result().template error<double>(),
                    neg_res.error<double>(), 1E-4);
    }
};

typedef ::testing::Types<
    aa::NoBinningAccumulator<double>,
    aa::LogBinningAccumulator<double>,
    aa::FullBinningAccumulator<double>
    > test_types;

TYPED_TEST_CASE(NegativeErrorTest, test_types);

TYPED_TEST(NegativeErrorTest, NegativeErrorTest) { this->negative_error_test(); }
