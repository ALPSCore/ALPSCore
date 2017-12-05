#include <alps/alea/mean.hpp>
#include <alps/alea/mpi.hpp>

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

template <typename Acc>
class mpi_twogauss_case
    : public ::testing::Test
{
public:
    typedef typename alps::alea::traits<Acc>::value_type value_type;

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
        alps::alea::mean_result<value_type> result = acc_.result();

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
