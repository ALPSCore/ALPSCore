/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/mc/mcbase.hpp>
#include <alps/stop_callback.hpp>

#include <boost/lambda/lambda.hpp>

// Simulation to measure e^(-x*x)
class my_sim_type : public alps::mcbase {

    public:

        my_sim_type(parameters_type const & params, std::size_t seed_offset = 42)
            : alps::mcbase(params, seed_offset)
            , total_count(params["COUNT"])

        {
            measurements << alps::accumulator::RealObservable("SValue")
                         << alps::accumulator::RealVectorObservable("VValue");
        }

        // if not compiled with mpi boost::mpi::communicator does not exists, 
        // so template the function
        template <typename Arg> my_sim_type(parameters_type const & params, Arg comm)
            : alps::mcbase(params, comm)
            , total_count(params["COUNT"])
        {
            measurements << alps::accumulator::RealObservable("SValue")
                         << alps::accumulator::RealVectorObservable("VValue");
        }

        // do the calculation in this function
        void update() {
            double x = random();
            value = exp(-x * x);
        };

        // do the measurements here
        void measure() {
            ++count;
            measurements["SValue"] << value;
            measurements["VValue"] << std::vector<double>(3, value);
        };

        double fraction_completed() const {
            return count / double(total_count);
        }

    private:
        int count;
        int total_count;
        double value;
};

int main(int argc, char *argv[]) {

    alps::mcoptions options(argc, argv);

    alps::parameters_type<my_sim_type>::type params;
    if (boost::filesystem::extension(options.input_file) == ".h5")
        alps::hdf5::archive(options.input_file)["/parameters"] >> params;
    else
        params = alps::parameters_type<my_sim_type>::type(options.input_file);

    my_sim_type my_sim(params); // creat a simulation
    my_sim.run(alps::stop_callback(options.time_limit)); // run the simulation

    alps::results_type<my_sim_type>::type results = collect_results(my_sim); // collect the results

    std::cout << "e^(-x*x): " << results["SValue"] << std::endl;
    std::cout << "e^(-x*x): " << results["VValue"] << std::endl;
    save_results(results, params, options.output_file, "/simulation/results");
}
