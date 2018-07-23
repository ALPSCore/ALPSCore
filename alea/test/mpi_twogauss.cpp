/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/alea/mpi.hpp>
#include <alps/alea/mean.hpp>
#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>
#include <alps/alea/autocorr.hpp>
#include <alps/alea/batch.hpp>

#include "alps/utilities/gtest_par_xml_output.hpp"
#include "gtest/gtest.h"
#include "dataset.hpp"

#include <iostream>

TEST(reducer, setup)
{
    alps::mpi::communicator comm;
    alps::alea::mpi_reducer red(comm, 0);
    alps::alea::reducer_setup setup = red.get_setup();

    // check consistency between reducer and setup
    EXPECT_EQ(comm.rank(), (int)setup.pos);
    EXPECT_EQ(comm.size(), (int)setup.count);
    EXPECT_EQ(comm.rank() == 0, setup.have_result);
}

TEST(reducer, get_max)
{
    alps::mpi::communicator comm;
    alps::alea::mpi_reducer red(comm, 0);
    alps::alea::reducer_setup setup = red.get_setup();

    EXPECT_EQ(setup.count, (unsigned)red.get_max(setup.pos) + 1);
}

TEST(autocorr, asymmetry)
{
    alps::alea::autocorr_acc<double> acc_(2);
    alps::alea::mpi_reducer red_(alps::mpi::communicator(), 0);
    alps::alea::reducer_setup setup = red_.get_setup();

    if (setup.count < 2)
        return;     // nothing

    EXPECT_LT(setup.count, 32u);
    size_t mask = setup.pos == 0 ? (1 << (setup.count - 1)) - 1 : (1 << setup.pos) - 1;
    size_t sel = setup.pos == 0 ? 0 : 1 << (setup.pos - 1);

    std::vector<double> curr(2);
    for (size_t i = 0; i != twogauss_count; ++i) {
        if ((i & mask) != sel)
            continue;
        //std::cerr << "T " << setup.pos << " " << i << std::endl;

        std::copy(twogauss_data[i], twogauss_data[i+1], curr.begin());
        acc_ << curr;
    }

    alps::alea::autocorr_result<double> result = acc_.result();
    result.reduce(red_);

    EXPECT_EQ(setup.have_result, result.valid());
    if (setup.have_result) {
        std::vector<double> obs_mean = result.mean();
        EXPECT_NEAR(obs_mean[0], twogauss_mean[0], 1e-6);
        EXPECT_NEAR(obs_mean[1], twogauss_mean[1], 1e-6);
    }
}

template <typename Acc>
class mpi_twogauss_case
    : public ::testing::Test
{
public:
    typedef typename alps::alea::traits<Acc>::value_type value_type;
    typedef typename alps::alea::traits<Acc>::result_type result_type;

    mpi_twogauss_case()
        : acc_(2)
        , red_(alps::mpi::communicator(), 0)
    {
        alps::alea::reducer_setup setup = red_.get_setup();

        std::vector<value_type> curr(2);
        for (size_t i = setup.pos; i < twogauss_count; i += setup.count) {
            std::copy(twogauss_data[i], twogauss_data[i+1], curr.begin());
            acc_ << curr;
        }
    }

    void test_mean()
    {
        result_type result = acc_.result();

        alps::alea::reducer_setup setup = red_.get_setup();
        result.reduce(red_);

        EXPECT_EQ(setup.have_result, result.valid());
        if (setup.have_result) {
            std::vector<value_type> obs_mean = result.mean();
            EXPECT_NEAR(obs_mean[0], twogauss_mean[0], 1e-6);
            EXPECT_NEAR(obs_mean[1], twogauss_mean[1], 1e-6);
        }
    }

private:
    Acc acc_;
    alps::alea::mpi_reducer red_;
};

typedef ::testing::Types<
      alps::alea::mean_acc<double>
    , alps::alea::var_acc<double>
    , alps::alea::cov_acc<double>
    , alps::alea::autocorr_acc<double>
    , alps::alea::batch_acc<double>
    > test_types;

TYPED_TEST_CASE(mpi_twogauss_case, test_types);

TYPED_TEST(mpi_twogauss_case, test_mean) { this->test_mean(); }

int main(int argc, char** argv)
{
   alps::mpi::environment env(argc, argv, false);

   // Changes command line arguments to avoid every test to write in same file
   alps::gtest_par_xml_output tweak;
   tweak(alps::mpi::communicator().rank(), argc, argv);
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
