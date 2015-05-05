/** @file main.cpp
    @brief alps::params cmd_params
*/

#include <iostream>
#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/parseargs.hpp>
#include <alps/mc/stop_callback.hpp>

// Simulation class
// We extend alps::mcbase, which is the base class of all Monte Carlo simulations.
// Typically one would put this in a separate cpp file. Since this is a basic
// example, we include everything needed in a single file to help beginners to
// get oriented.
class my_sim_type : public alps::mcbase {
    
    // The internal state of our simulation
    private:
        // The current MC step
        int count;
        // The total MC step to do
        int total_count;
        // The value calculated for the current MC step
        double value;

    public:

        // The constructor for our simulation
        // We always need the parameters and the seed as we need to pass it to
        // the alps::mcbase constructor. We also initialize count to 0
        // and total_count based on the value of the nSteps parameter
        my_sim_type(parameters_type const & params, std::size_t seed_offset = 42)
            : alps::mcbase(params, seed_offset)
            , total_count(params["nSteps"])
            , count(0)
        {
            measurements << alps::accumulators::FullBinningAccumulator<double>("X")
                         << alps::accumulators::FullBinningAccumulator<double>("X2");
        }

        // if not compiled with mpi boost::mpi::communicator does not exists, 
        // so template the function
//        template <typename Arg> my_sim_type(parameters_type const & params, Arg comm)
//            : alps::mcbase(params, comm)
//            , total_count(params["COUNT"])
//        {
//            measurements << alps::accumulators::FullBinningAccumulator<double>("SValue")
//                         << alps::accumulators::FullBinningAccumulator<std::vector<double> >("VValue");
//        }

        // This performs the actual calculate at each MC step.
        // In this example we simply take a value from a uniform distribution.
        void update() {
            value = drand48();
        };

        // This collects the measurements at each time step.
        void measure() {
            // Increase the count
            count++;
            // Collect the value and the value squared
            measurements["X"] << value;
            measurements["X2"] << value*value;
        };

        // This must return a number from 0.0 to 1.0 that says how much
        // of the simulation has been completed
        double fraction_completed() const {
            return count / double(total_count);
        }

};


/**
 * This example shows how to setup a simple simulation and retrieve some results.
 * The simulation just takes samples from a uniform distribution from 0.0 to 
 * 1.0, and estimates the average and the variance. Full binning
 * accumulators are used to collect X and X^2 at each MC step, and to
 * calculate the variance at the end of the simulation.
 * 
 * @param argc the number of arguments
 * @param argv the argument array
 * @return the exit code
 */
int main(int argc, const char* argv[])
{
    // Creates the parameters for the simulation
    std::cout << "Initializing parameters..." << std::endl;
    alps::parameters_type<my_sim_type>::type params(argc, argv);

    // Define the parameters for our simulation, including the ones for the
    // base class
    params.define("nSteps", 1000, "Number of MC steps to perform");
    my_sim_type::define_parameters(params);
    
    // Create and run the simulation
    std::cout << "Running simulation..." << std::endl;
    my_sim_type my_sim(params);
    my_sim.run(alps::stop_callback(5));

    // Collect the results from the simulation
    std::cout << "Collecting results..." << std::endl;
    alps::results_type<my_sim_type>::type results = collect_results(my_sim);

    // Print the mean and the standard deviation
    std::cout << "Results:" << std::endl;
    std::cout << " mean: " << results["X"] << std::endl;
    std::cout << " variance: " << results["X2"] - results["X"]*results["X"] << std::endl;

    return 0;
}
