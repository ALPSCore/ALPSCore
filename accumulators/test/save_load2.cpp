/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file saveload2.cpp: Tests for saving/loading accumulator. FIXME: to be combined with saveload.cpp */

// For remove()
#include <cstdio>

#include "alps/accumulators.hpp"
#include "gtest/gtest.h"
#include "accumulator_generator.hpp"


// Data generation functions

using namespace alps::accumulators::testing;

// to pass accumulator types
template <template<typename> class A, typename T, typename G=RandomData>
struct AccumulatorTypeGenerator {
    typedef AccResultGenerator<A<T>, /*NPOINTS*/ 50000, G> accumulator_gen_type;
};

template <typename G>
struct AccumulatorTest : public testing::Test {
    typedef typename G::accumulator_gen_type acc_gen_type;
    typedef typename acc_gen_type::named_acc_type acc_type;
    typedef typename acc_gen_type::value_type value_type;
    
    static const int NPOINTS=acc_gen_type::NPOINTS; 

    acc_gen_type acc_gen;

    // Ugly, but should work
    static const bool is_mean_acc=is_same_accumulator<acc_type, alps::accumulators::MeanAccumulator>::value;
    static const bool is_nobin_acc=is_same_accumulator<acc_type, alps::accumulators::NoBinningAccumulator>::value;
    

    /// Save accumulator
    void TestSaveAccumulator() {
        const std::string fname="save_acc.h5";
        std::remove(fname.c_str());
        const alps::accumulators::accumulator_set& m=acc_gen.accumulators();
        alps::hdf5::archive ar(fname,"w");
        ar["dataset"] << m;
    }

    /// Save result set
    void TestSaveResult() {
        const std::string fname="save_res.h5";
        std::remove(fname.c_str());
        const alps::accumulators::result_set& res=acc_gen.results();
        alps::hdf5::archive ar(fname,"w");
        ar["dataset"] << res;
    }

    /// Save and load accumulator set, check results
    void TestSaveLoadAccumulator() {
        const std::string fname="saveload_acc.h5";
        std::remove(fname.c_str());
        const alps::accumulators::accumulator_set& m=acc_gen.accumulators();
        {
            alps::hdf5::archive ar(fname,"w");
            ar["dataset"] << m;
        }
        alps::accumulators::accumulator_set m1;
        {
            alps::hdf5::archive ar(fname,"r");
            ar["dataset"] >> m1;
        }
        const alps::accumulators::result_set r(m);
        const alps::accumulators::result_set r1(m1);
        EXPECT_NEAR(r["data"].mean<value_type>(), r1["data"].mean<value_type>(), 1E-8);
        if (is_mean_acc) return;
        EXPECT_NEAR(r["data"].error<value_type>(), r1["data"].error<value_type>(), 1E-8);
        if (is_nobin_acc) return;
        EXPECT_NEAR(r["data"].autocorrelation<value_type>(), r1["data"].autocorrelation<value_type>(), 1E-8);
    }

    /// Save and load result set, check results
    void TestSaveLoadResult() {
        const std::string fname="saveload_res.h5";
        std::remove(fname.c_str());
        const alps::accumulators::result_set& r=acc_gen.results();
        {
            alps::hdf5::archive ar(fname,"w");
            ar["dataset"] << r;
        }
        alps::accumulators::result_set r1;
        {
            alps::hdf5::archive ar(fname,"r");
            ar["dataset"] >> r1;
        }
        EXPECT_NEAR(r["data"].mean<value_type>(), r1["data"].mean<value_type>(), 1E-8);
        if (is_mean_acc) return;
        EXPECT_NEAR(r["data"].error<value_type>() ,r1["data"].error<value_type>(), 1E-8);
        if (is_nobin_acc) return;
        EXPECT_NEAR(r["data"].autocorrelation<value_type>(), r1["data"].autocorrelation<value_type>(), 1E-8);
    }

};

typedef ::testing::Types<
    AccumulatorTypeGenerator<alps::accumulators::FullBinningAccumulator,double>
    ,AccumulatorTypeGenerator<alps::accumulators::LogBinningAccumulator,double>
    ,AccumulatorTypeGenerator<alps::accumulators::NoBinningAccumulator,double>
    ,AccumulatorTypeGenerator<alps::accumulators::MeanAccumulator, double>
    > MyTypes;

TYPED_TEST_CASE(AccumulatorTest, MyTypes);

#define MAKE_TEST(_name_) TYPED_TEST(AccumulatorTest, _name_)  { this->TestFixture::_name_(); }

MAKE_TEST(TestSaveAccumulator)
MAKE_TEST(TestSaveLoadAccumulator)
MAKE_TEST(TestSaveResult)
MAKE_TEST(TestSaveLoadResult)
