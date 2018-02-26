/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/*
  Test that Result<NonScalarT,...> has correct member type Result<ScalarT,...>
*/

#include "alps/accumulators.hpp"
#include "gtest/gtest.h"

#include <vector>
#include <typeinfo>

/// Provide a "scalar" and a "nonscalar" (here: std::vector) based on type T
template <typename T>
struct gen_vector_type {
    typedef T scalar_type;
    typedef std::vector<T> nonscalar_type;
};

using namespace alps::accumulators;

/// Provide all raw-Result types for a given type T
template <typename T>
struct result_type {
    typedef impl::ResultBase<T> result_base;
    typedef impl::Result<T, count_tag, result_base> count_type;
    // typedef impl::Result<T, mean_tag, count_type> mean_type;
    typedef typename MeanAccumulator<T>::accumulator_type::result_type mean_type;
    // typedef impl::Result<T, error_tag, mean_type> nobinning_type;
    typedef typename NoBinningAccumulator<T>::accumulator_type::result_type nobinning_type;
    // typedef impl::Result<T, binning_analysis_tag, nobinning_type> logbinning_type;
    typedef typename LogBinningAccumulator<T>::accumulator_type::result_type logbinning_type;
    // typedef impl::Result<T, max_num_binning_tag, logbinning_type> fullbinning_type;
    typedef typename FullBinningAccumulator<T>::accumulator_type::result_type fullbinning_type;
};
    
/// GoogleTest fixture: test that a Result<nonscalar,...> has corresponding Result<scalar,...>
template <typename G>
struct ScalarResultTypeTest: public ::testing::Test {
    typedef typename G::scalar_type scalar_type;
    typedef typename G::nonscalar_type nonscalar_type;

#define TestScalar(feature_)                                            \
    void TestScalarX ## feature_()                                      \
    {                                                                   \
        EXPECT_EQ(typeid(void), typeid(typename result_type<scalar_type>::feature_##_type::scalar_result_type)); \
    }

    TestScalar(count);
    TestScalar(mean);
    TestScalar(nobinning);
    TestScalar(logbinning);
    TestScalar(fullbinning);
#undef TestScalar

#define TestNonscalar(feature_)                                         \
    void TestNonscalarX ## feature_()                                   \
    {                                                                   \
        EXPECT_EQ(typeid(typename result_type<scalar_type>::feature_##_type), \
                  typeid(typename result_type<nonscalar_type>::feature_##_type::scalar_result_type)); \
    }

    TestNonscalar(count);
    TestNonscalar(mean);
    TestNonscalar(nobinning);
    TestNonscalar(logbinning);
    TestNonscalar(fullbinning);
#undef TestNonscalar    
};

typedef ::testing::Types< gen_vector_type<double> /*, gen_matrix_type<double> */ > MyTypes;
TYPED_TEST_CASE(ScalarResultTypeTest, MyTypes);

#define GenerateTest(feature_) \
    TYPED_TEST(ScalarResultTypeTest, TestScalarX##feature_) { this->TestScalarX##feature_(); } \
    TYPED_TEST(ScalarResultTypeTest, TestNonscalarX##feature_) { this->TestNonscalarX##feature_(); } 
    
GenerateTest(count);
GenerateTest(mean);
GenerateTest(nobinning);
GenerateTest(logbinning);
GenerateTest(fullbinning);

#undef GenerateTest
