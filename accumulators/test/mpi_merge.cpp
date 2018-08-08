/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** Test for MPI merge functionality */

#include <algorithm>

#include "alps/utilities/mpi.hpp"

#include "alps/config.hpp"
#include "alps/accumulators.hpp"

#include "alps/utilities/gtest_par_xml_output.hpp"
#include "gtest/gtest.h"




// Name for the observable
#define OBSNAME "value"
// Size for the observable vector
#define VECSIZE (3u)


// Service function: scalar data of value val
template <typename T>
static inline T get_data(double val, int, T*)
{
    return val;
}

// Service function: vector data of value val
template <typename T>
static inline std::vector<T> get_data(double val, int n, std::vector<T>*)
{
    return std::vector<T>(n,val);
}

// using Google Test Fixture
// Use Accumulator of type A, having scalar or vector  as its value type
template <typename A>
class AccumulatorTest : public ::testing::Test {
  public:
    typedef typename alps::accumulators::value_type<typename A::accumulator_type>::type value_type;

    // compare scalar values
    template <typename T>
    void compare_values(double expected, T val, const char* msg, int rank) const
    {
        EXPECT_NEAR(expected, val, 1.E-3) << "Scalar " << msg
                                          << " is incorrect in MPI rank " << rank;
    }

    // compare vector values
    template <typename T>
    void compare_values(double expected, const std::vector<T>& val, const char* msg, int rank) const
    {
        EXPECT_EQ(VECSIZE, val.size()) << "Vector " << msg
                                       << "size is incorrect in MPI rank " << rank;
        for (unsigned int i=0; i<VECSIZE; ++i) {
            EXPECT_NEAR(expected, val[i], 1.E-3) << "Vector " << msg
                                                 << " element #" << i
                                                 << " is incorrect in MPI rank " << rank;
        }
    }

    // the actual test: merging as a part of a set
    void TestInSet(const std::vector<unsigned>& nsamples, const alps::mpi::communicator& comm)
    {
        // Prepare data for each rank (note: value_type may be a vector<T>!)
        alps::accumulators::accumulator_set measurements;
        measurements << A(OBSNAME);
        alps::accumulators::accumulator_wrapper& acc=measurements[OBSNAME];
        const unsigned ns=nsamples[comm.rank()];
        srand48(43);
        for (unsigned int i=0; i<ns; ++i) {
            acc << get_data(drand48(), VECSIZE, (value_type*)0);
        }

        // merge data
        acc.collective_merge(comm, 0);

        if (comm.rank()==0) { // the accumulator is valid only on master
            // extract results
            alps::accumulators::result_set results(measurements);
            const alps::accumulators::result_wrapper& res=results[OBSNAME];

            // test results
            int ntot=std::accumulate(nsamples.begin(), nsamples.end(), 0);
            const double expected_mean=0.5;
            const double expected_err=1./(12*sqrt(ntot-1.0));

            compare_values(ntot, res.count(), "count", comm.rank());
            compare_values(expected_mean, res.mean<value_type>(), "mean", comm.rank());
            compare_values(expected_err, res.error<value_type>(), "error", comm.rank());
        }
    }

    // the same test for an individual accumulator
    void TestIndividual(const std::vector<unsigned>& nsamples, const alps::mpi::communicator& comm)
    {
        // Prepare data for each rank (note: value_type may be a vector<T>!)
        A acc(OBSNAME);
        const unsigned ns=nsamples[comm.rank()];
        srand48(43);
        for (unsigned int i=0; i<ns; ++i) {
            acc << get_data(drand48(), VECSIZE, (value_type*)0);
        }

        // merge data
        acc.collective_merge(comm, 0);

        if (comm.rank()==0) { // the accumulator is valid only on master
            // extract results
            const std::shared_ptr<alps::accumulators::result_wrapper> resptr=acc.result();

            // test results
            int ntot=std::accumulate(nsamples.begin(), nsamples.end(), 0);
            const double expected_mean=0.5;
            const double expected_err=1./(12*sqrt(ntot-1.0));
            compare_values(ntot, resptr->count(), "count", comm.rank());
            compare_values(expected_mean, resptr->mean<value_type>(), "mean", comm.rank());
            compare_values(expected_err, resptr->error<value_type>(), "error", comm.rank());
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

TYPED_TEST(AccumulatorTest, SetCollectResults)
{
    alps::mpi::communicator comm;
    std::vector<unsigned> nsamples(comm.size(), 100000); // 100000 samples for each rank
    this->TestInSet(nsamples, comm);
}

TYPED_TEST(AccumulatorTest, SingleCollectResults)
{
    alps::mpi::communicator comm;
    std::vector<unsigned> nsamples(comm.size(), 100000); // 100000 samples for each rank
    this->TestIndividual(nsamples,comm);
}


int main(int argc, char** argv)
{
   alps::mpi::environment env(argc, argv);
   alps::gtest_par_xml_output tweak;
   tweak(alps::mpi::communicator().rank(), argc, argv);
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
