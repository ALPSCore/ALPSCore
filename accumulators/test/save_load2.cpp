/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
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
    typedef typename acc_type::accumulator_type raw_accumulator_type;
    typedef typename acc_type::result_type raw_result_type;
    typedef typename acc_gen_type::value_type value_type;
    
    static const int NPOINTS=acc_gen_type::NPOINTS; 

    acc_gen_type acc_gen;

    // Ugly, but should work
    static const bool is_mean_acc=is_same_accumulator<acc_type, alps::accumulators::MeanAccumulator>::value;
    static const bool is_nobin_acc=is_same_accumulator<acc_type, alps::accumulators::NoBinningAccumulator>::value;
    

    /// Save accumulator
    void TestSaveAccumulator() {
        const std::string fname="sl_save_acc.h5";
        std::remove(fname.c_str());
        const alps::accumulators::accumulator_set& m=acc_gen.accumulators();
        alps::hdf5::archive ar(fname,"w");
        ar["dataset"] << m;
    }

    /// Save result set
    void TestSaveResult() {
        const std::string fname="sl_save_res.h5";
        std::remove(fname.c_str());
        const alps::accumulators::result_set& res=acc_gen.results();
        alps::hdf5::archive ar(fname,"w");
        ar["dataset"] << res;
    }

    /// Save and load accumulator set, check results
    void TestSaveLoadAccumulator() {
        const std::string fname="sl_saveload_acc.h5";
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

        // Check that the underlying accumulator holds the expected raw accumulator type
        m1["data"].extract<raw_accumulator_type>();
        
        const alps::accumulators::result_set r(m);
        const alps::accumulators::result_set r1(m1);
        EXPECT_EQ(r["data"].count(), r1["data"].count());
        // NOTE: we use EXPECT_EQ(), not EXPECT_NEAR(), as we expect binary identical values.
        EXPECT_EQ(r["data"].mean<value_type>(), r1["data"].mean<value_type>());
        if (is_mean_acc) return;
        EXPECT_EQ(r["data"].error<value_type>(), r1["data"].error<value_type>());
        if (is_nobin_acc) return;
        EXPECT_EQ(r["data"].autocorrelation<value_type>(), r1["data"].autocorrelation<value_type>());
    }

    /// Save and load result set, check results
    void TestSaveLoadResult() {
        const std::string fname="sl_saveload_res.h5";
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

        // Check that the underlying result holds the expected raw result type
        r1["data"].extract<raw_result_type>();

        EXPECT_EQ(r["data"].count(), r1["data"].count());
        // NOTE: we use EXPECT_EQ(), not EXPECT_NEAR(), as we expect binary identical values.
        EXPECT_EQ(r["data"].mean<value_type>(), r1["data"].mean<value_type>());
        if (is_mean_acc) return;
        EXPECT_EQ(r["data"].error<value_type>() ,r1["data"].error<value_type>());
        if (is_nobin_acc) return;
        EXPECT_EQ(r["data"].autocorrelation<value_type>(), r1["data"].autocorrelation<value_type>());
    }

};

typedef std::vector<double> vdouble;
typedef std::vector<long double> vldouble;
typedef std::vector<float> vfloat;

typedef ::testing::Types<
    AccumulatorTypeGenerator<alps::accumulators::FullBinningAccumulator,float>
    ,AccumulatorTypeGenerator<alps::accumulators::LogBinningAccumulator,float>
    ,AccumulatorTypeGenerator<alps::accumulators::NoBinningAccumulator, float>
    ,AccumulatorTypeGenerator<alps::accumulators::MeanAccumulator,      float>

    ,AccumulatorTypeGenerator<alps::accumulators::FullBinningAccumulator,double>
    ,AccumulatorTypeGenerator<alps::accumulators::LogBinningAccumulator, double>
    ,AccumulatorTypeGenerator<alps::accumulators::NoBinningAccumulator,  double>
    ,AccumulatorTypeGenerator<alps::accumulators::MeanAccumulator,       double>

    ,AccumulatorTypeGenerator<alps::accumulators::FullBinningAccumulator,long double>
    ,AccumulatorTypeGenerator<alps::accumulators::LogBinningAccumulator, long double>
    ,AccumulatorTypeGenerator<alps::accumulators::NoBinningAccumulator,  long double>
    ,AccumulatorTypeGenerator<alps::accumulators::MeanAccumulator,       long double>

    ,AccumulatorTypeGenerator<alps::accumulators::FullBinningAccumulator,vfloat>
    ,AccumulatorTypeGenerator<alps::accumulators::LogBinningAccumulator, vfloat>
    ,AccumulatorTypeGenerator<alps::accumulators::NoBinningAccumulator,  vfloat>
    ,AccumulatorTypeGenerator<alps::accumulators::MeanAccumulator,       vfloat>

    ,AccumulatorTypeGenerator<alps::accumulators::FullBinningAccumulator,vdouble>
    ,AccumulatorTypeGenerator<alps::accumulators::LogBinningAccumulator, vdouble>
    ,AccumulatorTypeGenerator<alps::accumulators::NoBinningAccumulator,  vdouble>
    ,AccumulatorTypeGenerator<alps::accumulators::MeanAccumulator,       vdouble>

#ifdef ALPS_HDF5_1_8 /* these tests fail with HDF5 1.10+ */
    ,AccumulatorTypeGenerator<alps::accumulators::FullBinningAccumulator,vldouble>
    ,AccumulatorTypeGenerator<alps::accumulators::LogBinningAccumulator, vldouble>
    ,AccumulatorTypeGenerator<alps::accumulators::NoBinningAccumulator,  vldouble>
    ,AccumulatorTypeGenerator<alps::accumulators::MeanAccumulator,       vldouble>
#endif
    > MyTypes;

TYPED_TEST_CASE(AccumulatorTest, MyTypes);

#define MAKE_TEST(_name_) TYPED_TEST(AccumulatorTest, _name_)  { this->TestFixture::_name_(); }

MAKE_TEST(TestSaveAccumulator)
MAKE_TEST(TestSaveLoadAccumulator)
MAKE_TEST(TestSaveResult)
MAKE_TEST(TestSaveLoadResult)
