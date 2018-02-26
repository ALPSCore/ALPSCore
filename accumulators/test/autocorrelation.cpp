/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "alps/accumulators.hpp"
#include "gtest/gtest.h"

#include <iostream> /* FIXME: will go away whith proper Tau testing replacing printouts */

// Google Test fixture: type T is the data type
// FIXME: make it over accumulator types rather than data types
template <typename T>
class AccumulatorAutocorrelationTest : public ::testing::Test {
  public:
    // typedef alps::accumulators::FullBinningAccumulator<T> named_acc_type;
    typedef alps::accumulators::LogBinningAccumulator<T> named_acc_type;
    typedef T value_type;
    typedef typename named_acc_type::accumulator_type acc_type;
    typedef typename alps::accumulators::autocorrelation_type<acc_type>::type autocorrelation_type;

    named_acc_type named_acc;

    // Fill the named accumulator with some data
    AccumulatorAutocorrelationTest() : named_acc("binning acc")
    {
        // FIXME: we need vector type data here too
        // FIXME: we need more data for sensible autocorrelation
        named_acc << T(0);
        named_acc << T(1);
    }
};

// FIXME! The test is incomplete (vector types are not tested)
typedef ::testing::Types < double,float > test_data_types;

TYPED_TEST_CASE(AccumulatorAutocorrelationTest, test_data_types);

TYPED_TEST(AccumulatorAutocorrelationTest, autocorrelation)
{
    // FIXME! The test is incomplete (does not actually test anything)
    typename TestFixture::autocorrelation_type tau=this->named_acc.tau();
    std::cout << "Autocorrelation tau=" << tau << std::endl;
}
