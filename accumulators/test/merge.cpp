/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"

#include "accumulator_generator.hpp"

/// Proxy template to pass accumulator A, data generator G, number of point N
template <typename A, typename G, std::size_t N1=1000, std::size_t N2=2000, int E=8>
struct generator {
    typedef A acc_type;
    typedef G gen_type;
    static const std::size_t NPOINTS1=N1;
    static const std::size_t NPOINTS2=N2;
    static const int TOL=E; // accuracy, as in 10^(-TOL)
};

namespace aa=alps::accumulators;
namespace aat=alps::accumulators::testing;

/// GTest fixture: parametrized over accumulators and data generators
template <typename G>
struct AccumulatorMergeTest : public ::testing::Test {
    typedef typename G::acc_type acc_type;
    typedef typename G::gen_type gen_type;
    typedef typename aa::value_type<typename acc_type::accumulator_type>::type value_type;
    static const std::size_t NPOINTS1=G::NPOINTS1;
    static const std::size_t NPOINTS2=G::NPOINTS2;
    static const bool is_mean_acc = aat::is_same_accumulator<acc_type, aa::MeanAccumulator>::value;

    static double tol() { return std::pow(10.,-G::TOL); }
    
    gen_type gen;
    aa::accumulator_set ms_half1, ms_half2, ms_full;

    AccumulatorMergeTest() : gen() {
        ms_half1 << acc_type("data");
        ms_half2 << acc_type("data");
        ms_full  << acc_type("data");
        
        for (std::size_t i=0; i<NPOINTS1; ++i) {
            value_type v=aat::gen_data<value_type>(gen());
            ms_half1["data"] << v;
            ms_full["data"] << v;
        }
            
        for (std::size_t i=0; i<NPOINTS2; ++i) {
            value_type v=aat::gen_data<value_type>(gen());
            ms_half2["data"] << v;
            ms_full["data"] << v;
        }

        ms_half1.merge(ms_half2);
    };

    void testCount() {
        EXPECT_EQ(ms_full["data"].count(), ms_half1["data"].count());
    }

    void testMean() {
        EXPECT_NEAR(ms_full["data"].template mean<value_type>(),
                    ms_half1["data"].template mean<value_type>(),
                    tol());
    }

    void testErrorBar() {
        if (is_mean_acc) return;
        EXPECT_NEAR(ms_full["data"].template error<value_type>(),
                    ms_half1["data"].template error<value_type>(),
                    tol());
    }
};

TYPED_TEST_CASE_P(AccumulatorMergeTest);

TYPED_TEST_P(AccumulatorMergeTest, Count)    { this->TestFixture::testCount(); }
TYPED_TEST_P(AccumulatorMergeTest, Mean)     { this->TestFixture::testMean(); }
TYPED_TEST_P(AccumulatorMergeTest, ErrorBar) { this->TestFixture::testErrorBar(); }

REGISTER_TYPED_TEST_CASE_P(AccumulatorMergeTest,
                           Count, Mean, ErrorBar);

typedef ::testing::Types<
    // generator<aa::FullBinningAccumulator<double>, aat::ConstantData, 1000, 1000>,
    // generator<aa::FullBinningAccumulator<double>, aat::ConstantData, 1000, 2000>,
    // generator<aa::FullBinningAccumulator<double>, aat::ConstantData, 2000, 1000>,

    // generator<aa::FullBinningAccumulator<double>, aat::AlternatingData, 1000, 1000>,
    // generator<aa::FullBinningAccumulator<double>, aat::AlternatingData, 2000, 1000>,
    // generator<aa::FullBinningAccumulator<double>, aat::AlternatingData, 1000, 2000>,

    // generator<aa::FullBinningAccumulator<double>, aat::RandomData, 1000, 1000, 4>,
    // generator<aa::FullBinningAccumulator<double>, aat::RandomData, 1000, 3000, 4>,
    // generator<aa::FullBinningAccumulator<double>, aat::RandomData, 3000, 1000, 4>,
    
    // generator<aa::FullBinningAccumulator<double>, aat::CorrelatedData<5>, 1000, 1000, 3>,
    // generator<aa::FullBinningAccumulator<double>, aat::CorrelatedData<5>, 2000, 1000, 3>,
    // generator<aa::FullBinningAccumulator<double>, aat::CorrelatedData<5>, 1000, 2000, 3>,

    generator<aa::LogBinningAccumulator<double>, aat::ConstantData, 1000, 1000>,
    generator<aa::LogBinningAccumulator<double>, aat::ConstantData, 1000, 2000>,
    generator<aa::LogBinningAccumulator<double>, aat::ConstantData, 2000, 1000>,

    generator<aa::LogBinningAccumulator<double>, aat::AlternatingData, 1000, 1000>,
    generator<aa::LogBinningAccumulator<double>, aat::AlternatingData, 2000, 1000>,
    generator<aa::LogBinningAccumulator<double>, aat::AlternatingData, 1000, 2000>,
    
    // generator<aa::LogBinningAccumulator<double>, aat::LinearData, 1000, 1000>,
    // generator<aa::LogBinningAccumulator<double>, aat::LinearData, 1000, 2000>,
    // generator<aa::LogBinningAccumulator<double>, aat::LinearData, 2000, 1000>,
    
    generator<aa::LogBinningAccumulator<double>, aat::CorrelatedData<5>, 1000, 1000, 3>,
    generator<aa::LogBinningAccumulator<double>, aat::CorrelatedData<5>, 2000, 1000, 3>,
    generator<aa::LogBinningAccumulator<double>, aat::CorrelatedData<5>, 1000, 2000, 3>,

    generator<aa::LogBinningAccumulator<double>, aat::RandomData, 1000, 1000, 4>,
    generator<aa::LogBinningAccumulator<double>, aat::RandomData, 1000, 3000, 4>,
    generator<aa::LogBinningAccumulator<double>, aat::RandomData, 3000, 1000, 4>,
    
    generator<aa::MeanAccumulator<double>      , aat::LinearData>,
    generator<aa::MeanAccumulator<double>      , aat::CorrelatedData<5> >,
    generator<aa::MeanAccumulator<double>      , aat::RandomData>,
    generator<aa::NoBinningAccumulator<double> , aat::LinearData>,
    generator<aa::NoBinningAccumulator<double> , aat::CorrelatedData<5> >,
    generator<aa::NoBinningAccumulator<double> , aat::RandomData>
    > MyTypes;

INSTANTIATE_TYPED_TEST_CASE_P(test1, AccumulatorMergeTest, MyTypes);
