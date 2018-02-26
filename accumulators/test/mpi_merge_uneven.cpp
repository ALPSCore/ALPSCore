/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** Test for MPI merge functionality for chains with different number of measurements */

#include <algorithm>

// #include "alps/utilities/mpi.hpp"
#include "alps/accumulators.hpp"

#include "alps/utilities/gtest_par_xml_output.hpp"
#include "gtest/gtest.h"

#include "accumulator_generator.hpp"


// how much th enumber of measurements differ between ranks:
#define COUNT_RANK0 1000
#define COUNT_DIFF_PER_RANK 5000

// test vector size
#define VECSIZE 3

namespace aac=alps::accumulators;
namespace aact=aac::testing;

// using Google Test Fixture parametrized over named accumulator type A
template <typename A>
class AccumulatorTest : public ::testing::Test {
  public:
    typedef A accumulator_type;
    typedef typename aac::value_type<typename A::accumulator_type>::type value_type;
    typedef aact::ConstantData data_gen_type;
    static const int master=0;

    data_gen_type gen;
    accumulator_type acc;
    alps::mpi::communicator comm;
    int rank;
    unsigned int npoints; //< number of points in this rank
    unsigned int npoints_all; //< number of points in all ranks

    AccumulatorTest(): gen(1.0), acc("data"), comm(), rank(comm.rank())
    {
        int np=comm.size();
        npoints=COUNT_RANK0 + COUNT_DIFF_PER_RANK*rank;
        npoints_all=np*(2*COUNT_RANK0 + COUNT_DIFF_PER_RANK*(np-1))/2;
    }

    void merge_test()
    {
        for (unsigned int i=0; i<npoints; ++i) {
            acc << aact::gen_data<value_type>(gen(),VECSIZE).value();
        }
        acc.collective_merge(comm, master);
        if (master==rank) {
            // extract results
            const boost::shared_ptr<aac::result_wrapper> resptr=acc.result();
            const aac::result_wrapper& res=*resptr;

            value_type expected_mean=aact::gen_data<value_type>(gen.mean(npoints_all), VECSIZE);
            value_type expected_error=aact::gen_data<value_type>(gen.error(npoints_all), VECSIZE);
            EXPECT_EQ(npoints_all, res.count());
            aact::compare_near(expected_mean, res.mean<value_type>(), 1E-5, "mean value");
            aact::compare_near(expected_error, res.error<value_type>(), 1E-5, "errorbar value");
        }
    }
};

typedef std::vector<double> doublevec;
typedef std::vector<float> floatvec;

typedef ::testing::Types<
    alps::accumulators::NoBinningAccumulator<double>,
    alps::accumulators::LogBinningAccumulator<double>,
    alps::accumulators::FullBinningAccumulator<double>,

    alps::accumulators::NoBinningAccumulator<float>,
    alps::accumulators::LogBinningAccumulator<float>,
    alps::accumulators::FullBinningAccumulator<float>,

    alps::accumulators::NoBinningAccumulator<doublevec>,
    alps::accumulators::LogBinningAccumulator<doublevec>,
    alps::accumulators::FullBinningAccumulator<doublevec>,

    alps::accumulators::NoBinningAccumulator<floatvec>,
    alps::accumulators::LogBinningAccumulator<floatvec>,
    alps::accumulators::FullBinningAccumulator<floatvec>
    > MyTypes;

TYPED_TEST_CASE(AccumulatorTest, MyTypes);

TYPED_TEST(AccumulatorTest, merge_test) { this->merge_test(); }

int main(int argc, char** argv)
{
   alps::mpi::environment env(argc, argv);
   alps::gtest_par_xml_output tweak;
   tweak(alps::mpi::communicator().rank(), argc, argv);
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
