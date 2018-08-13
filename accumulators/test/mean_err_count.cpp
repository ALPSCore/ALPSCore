/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file mean_err_count.cpp
    Test basic accumulator statistics.
*/

#include <boost/math/special_functions/fpclassify.hpp> /* for portable isinf() */

#include "alps/accumulators.hpp"

#include "gtest/gtest.h"

#include "../../utilities/test/vector_comparison_predicates.hpp" /* FIXME: relative path is fragile!! */

#include "accumulator_generator.hpp"

namespace aa=alps::accumulators;
namespace aat=aa::testing;
namespace at=alps::testing;

// enum test_errbar_flag_type { TEST_ERRBAR, DONT_TEST_ERRBAR };

// handy type shortcuts
typedef std::vector<float> v_float;
typedef std::vector<double> v_double;
typedef std::vector<long double> v_ldouble;

// Proxy template to pass accumulator type, generator, number of tests, and tolerance as 10^-E
template <typename A, typename G, std::size_t N, int E=8>
struct generator {
    typedef aat::AccResultGenerator<A,N,G> generator_type;
    static const int TOLEXP=E;
};

// GoogleTest fixture parametrized over data generator, via the proxy template above
template <typename G>
struct AccumulatorStatTest : public testing::Test {
    typedef typename G::generator_type generator_type;
    typedef typename generator_type::named_acc_type acc_type;
    typedef typename generator_type::value_type value_type;

    static const std::size_t NPOINTS=generator_type::NPOINTS;
    static const bool is_mean_acc=aat::is_same_accumulator<acc_type, aa::MeanAccumulator>::value;

    static double tol() { return std::pow(10.,-G::TOLEXP); }

    value_type expected_err() { return aat::gen_data<value_type>(gen.expected_err()); }
    value_type expected_mean() { return aat::gen_data<value_type>(gen.expected_mean()); }

    generator_type gen;

    void testCount() {
        std::size_t ac=count(gen.accumulator());
        EXPECT_EQ(NPOINTS+0, ac) << "Accumulator count differs";
        std::size_t rc=count(gen.accumulator());
        EXPECT_EQ(NPOINTS+0, rc) << "Result count differs";
    }

    void testMean() {
        value_type amean=gen.accumulator().template mean<value_type>();
        // aat::compare_near(expected_mean(), amean, tol(), "Accumulator mean");
	EXPECT_TRUE(at::is_near(expected_mean(), amean, tol())) << "Accumulator mean";

        value_type rmean=gen.result().template mean<value_type>();
        // aat::compare_near(expected_mean(), rmean, tol(), "Result mean");
	EXPECT_TRUE(at::is_near(expected_mean(), rmean, tol())) << "Result mean";
    }

    // This "naive" test is correct only for non-binning accumulators or for a large random data stream
    void testError() {
        if (is_mean_acc) {
#if defined(__APPLE__) && defined(__INTEL_COMPILER)
            //we were testing fore exceptions here. OSX/ICPC fails when doing this.
            EXPECT_ANY_THROW( value_type aerr=gen.accumulator().template error<value_type>() );
            EXPECT_ANY_THROW( value_type rerr=gen.result().template error<value_type>() );
#endif
            return;
        }
        value_type aerr=gen.accumulator().template error<value_type>();
        // aat::compare_near(expected_err(), aerr, tol(), "Accumulator error bar");
	EXPECT_TRUE(at::is_near(expected_err(), aerr, tol())) << "Accumulator error bar";

        value_type rerr=gen.result().template error<value_type>();
        // aat::compare_near(expected_err(), rerr, tol(), "Result error bar");
	EXPECT_TRUE(at::is_near(expected_err(), rerr, tol())) << "Result error bar";
    }
};

// Test set for "naive" error testing.

TYPED_TEST_CASE_P(AccumulatorStatTest);

#define DECLARE_TEST(_name_) TYPED_TEST_P(AccumulatorStatTest, _name_) { this->TestFixture::_name_(); }
DECLARE_TEST(testCount);
DECLARE_TEST(testMean);
DECLARE_TEST(testError);
#undef DECLARE_TEST

REGISTER_TYPED_TEST_CASE_P(AccumulatorStatTest, testCount, testMean, testError);


typedef ::testing::Types<
    generator<aa::MeanAccumulator<double>, aat::ConstantData, 2>,
    generator<aa::MeanAccumulator<double>, aat::ConstantData, 100>,
    generator<aa::MeanAccumulator<double>, aat::ConstantData, 20000>,
    generator<aa::MeanAccumulator<double>, aat::AlternatingData, 2>,
    generator<aa::MeanAccumulator<double>, aat::AlternatingData, 100>,
    generator<aa::MeanAccumulator<double>, aat::AlternatingData, 20000>,
    generator<aa::MeanAccumulator<double>, aat::LinearData, 2>,
    generator<aa::MeanAccumulator<double>, aat::LinearData, 100>,
    generator<aa::MeanAccumulator<double>, aat::LinearData, 20000>,
    generator<aa::MeanAccumulator<double>, aat::RandomData, 100000, 3>,

    generator<aa::MeanAccumulator<v_double>, aat::ConstantData, 2>,
    generator<aa::MeanAccumulator<v_double>, aat::ConstantData, 100>,
    generator<aa::MeanAccumulator<v_double>, aat::ConstantData, 20000>,
    generator<aa::MeanAccumulator<v_double>, aat::AlternatingData, 2>,
    generator<aa::MeanAccumulator<v_double>, aat::AlternatingData, 100>,
    generator<aa::MeanAccumulator<v_double>, aat::AlternatingData, 20000>,
    generator<aa::MeanAccumulator<v_double>, aat::LinearData, 2>,
    generator<aa::MeanAccumulator<v_double>, aat::LinearData, 100>,
    generator<aa::MeanAccumulator<v_double>, aat::LinearData, 20000>,
    generator<aa::MeanAccumulator<v_double>, aat::RandomData, 100000, 3>
    > double_mean_types;
INSTANTIATE_TYPED_TEST_CASE_P(DoubleMean, AccumulatorStatTest, double_mean_types);

typedef ::testing::Types<
    generator<aa::MeanAccumulator<float>, aat::ConstantData, 2>,
    generator<aa::MeanAccumulator<float>, aat::ConstantData, 100>,
    generator<aa::MeanAccumulator<float>, aat::ConstantData, 20000>,
    generator<aa::MeanAccumulator<float>, aat::AlternatingData, 2>,
    generator<aa::MeanAccumulator<float>, aat::AlternatingData, 100>,
    generator<aa::MeanAccumulator<float>, aat::AlternatingData, 20000>,
    generator<aa::MeanAccumulator<float>, aat::LinearData, 2>,
    generator<aa::MeanAccumulator<float>, aat::LinearData, 100>,
    // generator<aa::MeanAccumulator<float>, aat::LinearData, 10000>, 
    generator<aa::MeanAccumulator<float>, aat::RandomData, 100000, 3>,

    generator<aa::MeanAccumulator<v_float>, aat::ConstantData, 2>,
    generator<aa::MeanAccumulator<v_float>, aat::ConstantData, 100>,
    generator<aa::MeanAccumulator<v_float>, aat::ConstantData, 20000>,
    generator<aa::MeanAccumulator<v_float>, aat::AlternatingData, 2>,
    generator<aa::MeanAccumulator<v_float>, aat::AlternatingData, 100>,
    generator<aa::MeanAccumulator<v_float>, aat::AlternatingData, 20000>,
    generator<aa::MeanAccumulator<v_float>, aat::LinearData, 2>,
    generator<aa::MeanAccumulator<v_float>, aat::LinearData, 100>,
    // generator<aa::MeanAccumulator<v_float>, aat::LinearData, 10000>,
    generator<aa::MeanAccumulator<v_float>, aat::RandomData, 100000, 3>
    > float_mean_types;
INSTANTIATE_TYPED_TEST_CASE_P(FloatMean, AccumulatorStatTest, float_mean_types);

typedef ::testing::Types<
    generator<aa::NoBinningAccumulator<double>, aat::ConstantData, 2>,
    generator<aa::NoBinningAccumulator<double>, aat::ConstantData, 100>,
    generator<aa::NoBinningAccumulator<double>, aat::ConstantData, 20000>,
    generator<aa::NoBinningAccumulator<double>, aat::AlternatingData, 2>,
    generator<aa::NoBinningAccumulator<double>, aat::AlternatingData, 100>,
    generator<aa::NoBinningAccumulator<double>, aat::AlternatingData, 20000>,
    generator<aa::NoBinningAccumulator<double>, aat::LinearData, 2>,
    generator<aa::NoBinningAccumulator<double>, aat::LinearData, 100>,
    generator<aa::NoBinningAccumulator<double>, aat::LinearData, 20000>,
    generator<aa::NoBinningAccumulator<double>, aat::RandomData, 100000, 3>,

    generator<aa::NoBinningAccumulator<v_double>, aat::ConstantData, 2>,
    generator<aa::NoBinningAccumulator<v_double>, aat::ConstantData, 100>,
    generator<aa::NoBinningAccumulator<v_double>, aat::ConstantData, 20000>,
    generator<aa::NoBinningAccumulator<v_double>, aat::AlternatingData, 2>,
    generator<aa::NoBinningAccumulator<v_double>, aat::AlternatingData, 100>,
    generator<aa::NoBinningAccumulator<v_double>, aat::AlternatingData, 20000>,
    generator<aa::NoBinningAccumulator<v_double>, aat::LinearData, 2>,
    generator<aa::NoBinningAccumulator<v_double>, aat::LinearData, 100>,
    generator<aa::NoBinningAccumulator<v_double>, aat::LinearData, 20000>,
    generator<aa::NoBinningAccumulator<v_double>, aat::RandomData, 100000, 3>
    > double_nobin_types;
INSTANTIATE_TYPED_TEST_CASE_P(DoubleNobin, AccumulatorStatTest, double_nobin_types);

typedef ::testing::Types<
    generator<aa::NoBinningAccumulator<float>, aat::ConstantData, 2>,
    generator<aa::NoBinningAccumulator<float>, aat::ConstantData, 100>,
    generator<aa::NoBinningAccumulator<float>, aat::ConstantData, 20000>,
    generator<aa::NoBinningAccumulator<float>, aat::AlternatingData, 2>,
    generator<aa::NoBinningAccumulator<float>, aat::AlternatingData, 100>,
    generator<aa::NoBinningAccumulator<float>, aat::AlternatingData, 20000>,
    generator<aa::NoBinningAccumulator<float>, aat::LinearData, 2>,
    generator<aa::NoBinningAccumulator<float>, aat::LinearData, 100>,
    // generator<aa::NoBinningAccumulator<float>, aat::LinearData, 10000>, 
    generator<aa::NoBinningAccumulator<float>, aat::RandomData, 100000, 3>,

    generator<aa::NoBinningAccumulator<v_float>, aat::ConstantData, 2>,
    generator<aa::NoBinningAccumulator<v_float>, aat::ConstantData, 100>,
    generator<aa::NoBinningAccumulator<v_float>, aat::ConstantData, 20000>,
    generator<aa::NoBinningAccumulator<v_float>, aat::AlternatingData, 2>,
    generator<aa::NoBinningAccumulator<v_float>, aat::AlternatingData, 100>,
    generator<aa::NoBinningAccumulator<v_float>, aat::AlternatingData, 20000>,
    generator<aa::NoBinningAccumulator<v_float>, aat::LinearData, 2>,
    generator<aa::NoBinningAccumulator<v_float>, aat::LinearData, 100>,
    // generator<aa::NoBinningAccumulator<v_float>, aat::LinearData, 10000>,
    generator<aa::NoBinningAccumulator<v_float>, aat::RandomData, 100000, 3>
    > float_nobin_types;
INSTANTIATE_TYPED_TEST_CASE_P(FloatNobin, AccumulatorStatTest, float_nobin_types);

typedef ::testing::Types<
    generator<aa::LogBinningAccumulator<double>, aat::RandomData, 100000, 3>,
    generator<aa::FullBinningAccumulator<double>, aat::RandomData, 100000, 3>,

    generator<aa::LogBinningAccumulator<v_double>, aat::RandomData, 100000, 3>,
    generator<aa::FullBinningAccumulator<v_double>, aat::RandomData, 100000, 3>
    > double_random_bin_types;
INSTANTIATE_TYPED_TEST_CASE_P(DoubleLogbin, AccumulatorStatTest, double_random_bin_types);


// Now, another set of tests that expect error bar to be infinite
template <typename G>
struct AccumulatorStatInfErrTest : public AccumulatorStatTest<G> {
    typedef AccumulatorStatTest<G> base_type;
    typedef typename base_type::value_type value_type;
    
    template <typename T>
    static bool is_inf(const T& val) { return (boost::math::isinf)(val); }
    
    template <typename T>
    static bool is_inf(const std::vector<T>& val) {
        EXPECT_FALSE(val.empty()) << "Error vector is empty!!";
        if (val.empty()) return false;
        for(const T& elem: val) {
            if (!is_inf(elem)) return false;
        }
        return true;
    }

    
    void testError() {
        value_type rerr=this->gen.result().template error<value_type>();
        EXPECT_TRUE(is_inf(rerr)) << "Result error bar is incorrect";
    }
};

TYPED_TEST_CASE_P(AccumulatorStatInfErrTest);

#define DECLARE_TEST(_name_) TYPED_TEST_P(AccumulatorStatInfErrTest, _name_) { this->TestFixture::_name_(); }
DECLARE_TEST(testCount);
DECLARE_TEST(testMean);
DECLARE_TEST(testError);
#undef DECLARE_TEST

REGISTER_TYPED_TEST_CASE_P(AccumulatorStatInfErrTest, testCount, testMean, testError);


typedef ::testing::Types<
    generator<aa::NoBinningAccumulator<double>, aat::ConstantData, 1>,
    generator<aa::LogBinningAccumulator<double>, aat::ConstantData, 1>,
    generator<aa::FullBinningAccumulator<double>, aat::ConstantData, 1>,

    generator<aa::LogBinningAccumulator<double>, aat::ConstantData, 100>,
    generator<aa::LogBinningAccumulator<double>, aat::AlternatingData, 100>,
    generator<aa::LogBinningAccumulator<double>, aat::LinearData, 100>
    > double_short_types;

INSTANTIATE_TYPED_TEST_CASE_P(DoubleShort, AccumulatorStatInfErrTest, double_short_types);

typedef ::testing::Types<
    generator<aa::NoBinningAccumulator<v_double>, aat::ConstantData, 1>,
    generator<aa::LogBinningAccumulator<v_double>, aat::ConstantData, 1>,
    generator<aa::FullBinningAccumulator<v_double>, aat::ConstantData, 1>,

    generator<aa::LogBinningAccumulator<v_double>, aat::ConstantData, 100>,
    generator<aa::LogBinningAccumulator<v_double>, aat::AlternatingData, 100>,
    generator<aa::LogBinningAccumulator<v_double>, aat::LinearData, 100>
    > doublevec_short_types;

INSTANTIATE_TYPED_TEST_CASE_P(DoubleVectorShort, AccumulatorStatInfErrTest, doublevec_short_types);

