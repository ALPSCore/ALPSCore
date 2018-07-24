/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file repeated_merge.cpp

    Test behavior of repeated accumulators merges.
*/


#include <alps/config.hpp>
#include <alps/accumulators.hpp>

#include <alps/utilities/gtest_par_xml_output.hpp>
#include "gtest/gtest.h"

#include "./accumulator_generator.hpp"

namespace aa=alps::accumulators;
namespace aat=alps::accumulators::testing;

namespace {
    // Proxy template
    template <template<typename> class A, typename T=double>
    struct proxy {
        typedef T value_type;
        typedef A<value_type> accumulator_type;
    };
}

// Parameterized over the proxy containing named accumulator and its value type
template <typename P>
class AccumulatorTest : public ::testing::Test {
  public:
    typedef typename P::accumulator_type accumulator_type;
    typedef typename P::value_type value_type;
    // typedef aat::LinearData data_gen_type;
    typedef aat::RandomData data_gen_type;
    // This fails autocorrelation test!
    // // typedef aat::CorrelatedData<20> data_gen_type;
    int root_;
    alps::mpi::communicator comm_;
    int rank;
    bool is_master;
    data_gen_type gen_;
    static const unsigned int NSTEPS=10000;
    AccumulatorTest(): root_(0), comm_(), rank(comm_.rank()), is_master(rank==root_), gen_(rank) {}

    void twoMerges() {
        aa::accumulator_set aset;
        aset << accumulator_type("value")
             << accumulator_type("test_value");

        // Run the steps sequentially
        if (is_master) {
            for (int mock_rank=0; mock_rank<comm_.size(); ++mock_rank) {
                data_gen_type gen2(mock_rank);
                for (unsigned int i=0; i<2*NSTEPS; ++i) {
                    aset["test_value"] << gen2();
                }
            }
        }

        // Run the steps
        for (unsigned int i=0; i<NSTEPS; ++i) {
            aset["value"] << gen_();
        }
        {
            aa::accumulator_wrapper& acc=aset["value"];
            acc.collective_merge(comm_, root_);
            // This is done by collective_merge(...):
            // if (!is_master) acc.reset();
        }

        // Run again
        for (unsigned int i=0; i<NSTEPS; ++i) {
            aset["value"] << gen_();
        }

        {
            aa::accumulator_wrapper& acc=aset["value"];
            acc.collective_merge(comm_, root_);
        }

        if (is_master) {
            aa::result_set rset(aset);
            // ASSERT_EQ(2*NSTEPS*comm_.size(), rset["test_value"].count()) << "Sequential count";
            // ASSERT_NEAR(gen_.mean(2*NSTEPS), rset["test_value"].mean<value_type>(), 1E-5) << "Sequential Mean value";

            EXPECT_EQ(rset["test_value"].count(), rset["value"].count()) << "Count";
            EXPECT_NEAR(rset["test_value"].mean<value_type>(), rset["value"].mean<value_type>(), 1E-2) << "Mean value";

            if (!aat::is_same_accumulator<accumulator_type,aa::MeanAccumulator>::value) {
                EXPECT_NEAR(rset["test_value"].error<value_type>(), rset["value"].error<value_type>(), 1E-2) << "Error bar";
            }

            if (aat::is_same_accumulator<accumulator_type,aa::LogBinningAccumulator>::value ||
                aat::is_same_accumulator<accumulator_type,aa::FullBinningAccumulator>::value) {
                EXPECT_NEAR(rset["test_value"].autocorrelation<value_type>(),
                            rset["value"].autocorrelation<value_type>(), 5.00)
                    << "Autocorrelation value";
            }
        }
    }
};

typedef ::testing::Types<
    proxy<aa::MeanAccumulator>
    ,
    proxy<aa::NoBinningAccumulator>
    ,
    proxy<aa::LogBinningAccumulator>
    ,
    proxy<aa::FullBinningAccumulator>
    > namedAccumulatorTypes;

TYPED_TEST_CASE(AccumulatorTest, namedAccumulatorTypes);

TYPED_TEST(AccumulatorTest, twoMerges) { this->twoMerges(); }




int main(int argc, char** argv)
{
   alps::mpi::environment env(argc, argv, false);
   alps::gtest_par_xml_output tweak;
   tweak(alps::mpi::communicator().rank(), argc, argv);
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
