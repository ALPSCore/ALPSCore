/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <algorithm>

#include <alps/mc/mcbase.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mpiadapter.hpp>
#include <alps/mc/stop_callback.hpp>

#include <alps/testing/unique_file.hpp>

#include <boost/lambda/lambda.hpp>

#include "gtest/gtest.h"

// Simulation to measure e^(-x*x)
class my_sim_type : public alps::mcbase {

    public:

        static const unsigned int VSIZE=3;

        typedef std::vector<double> double_vector_type;

        void init()
        {
            measurements << alps::accumulators::FullBinningAccumulator<double>("SValue")
                         << alps::accumulators::FullBinningAccumulator<double_vector_type>("VValue")
                         << alps::accumulators::NoBinningAccumulator<double_vector_type>("VValue1");
        }

        my_sim_type(parameters_type const & params, std::size_t seed_offset = 42)
            : alps::mcbase(params, seed_offset)
            , total_count(params["COUNT"])
            , count(0)
            , value(0)

        {
            init();
        }

        // if not compiled with mpi alps::mpi::communicator does not exists,
        // so template the function
        template <typename Arg> my_sim_type(parameters_type const & params, Arg comm)
            : alps::mcbase(params, comm)
            , total_count(params["COUNT"])
            , count(0)
            , value(0)
        {
            init();
        }

        // do the calculation in this function
        void update() {
            double x = random();
            value = 1+x;
        };

        // do the measurements here
        void measure() {
            ++count;
            measurements["SValue"] << value;
            measurements["VValue"] << double_vector_type(VSIZE, value);
            measurements["VValue1"] << double_vector_type(VSIZE, value);
        };

        double fraction_completed() const {
            return count / double(total_count);
        }

    private:
        int total_count;
        int count;
        double value;
};

TEST(mc, sum_mpi){
        alps::mpi::communicator c;

        alps::mcbase::parameters_type params;
        const int maxcount=1000;
        params["COUNT"]=maxcount;
        my_sim_type::define_parameters(params); // do parameters definitions

        params.broadcast(c, 0);

        int t_min_check=1, t_max_check=1, timelimit=300;

        alps::mcmpiadapter<my_sim_type> my_sim(params, c, alps::check_schedule(t_min_check, t_max_check)); // create a simulation

        my_sim.run(alps::stop_callback(c, timelimit)); // run the simulation

        using alps::collect_results;

        if (c.rank() == 0) { // print the results and save it to hdf5
            alps::results_type<alps::mcmpiadapter<my_sim_type> >::type results = collect_results(my_sim);
            std::cout << "1+x: " << results["SValue"] << std::endl;
            std::cout << "1+x: " << results["VValue"] << std::endl;

            const double expected_mean=1.5;
            const double expected_err=(1./12)/sqrt(results["SValue"].count()-1);
            EXPECT_NEAR(expected_mean, results["SValue"].mean<double>(), 1.E-2) << "Scalar (FullBinning) mean is incorrect";
            EXPECT_NEAR(expected_err, results["SValue"].error<double>(), 1.E-2) << "Scalar (FullBinning) error is incorrect";

            // Test using NoBinningAccumulator
            {
                const my_sim_type::double_vector_type& vv_mean=results["VValue1"].mean<my_sim_type::double_vector_type>();
                const my_sim_type::double_vector_type& vv_err=results["VValue1"].error<my_sim_type::double_vector_type>();
                EXPECT_EQ(my_sim_type::VSIZE+0, vv_mean.size());
                for (unsigned int i=0; i<my_sim_type::VSIZE; ++i) {
                    EXPECT_NEAR(expected_mean, vv_mean[i], 1.E-2) << "Vector (NoBinning) mean is incorrect at #" << i;
                    EXPECT_NEAR(expected_err, vv_err[i], 1.E-2)  << "Vector (NoBinning) error is incorrect at #" << i;
                }
            }

            // Test using FullBinningAccumulator
            {
                const my_sim_type::double_vector_type& vv_mean=results["VValue"].mean<my_sim_type::double_vector_type>();
                const my_sim_type::double_vector_type& vv_err=results["VValue"].error<my_sim_type::double_vector_type>();
                EXPECT_EQ(my_sim_type::VSIZE+0, vv_mean.size());
                for (unsigned int i=0; i<my_sim_type::VSIZE; ++i) {
                    EXPECT_NEAR(expected_mean, vv_mean[i], 1.E-2)  << "Vector (FullBinning) mean is incorrect at #" << i;
                    EXPECT_NEAR(expected_err, vv_err[i], 1.E-2) << "Vector (FullBinning) error is incorrect at #" << i;
                }
            }

            alps::save_results(results, params, alps::testing::temporary_filename("sum_mpi.h5.") , "/simulation/results");
            EXPECT_TRUE(results["SValue"].count() >= maxcount);
        } else
            collect_results(my_sim);
}

int main(int argc, char** argv)
{
   alps::mpi::environment env(argc, argv, false);
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
