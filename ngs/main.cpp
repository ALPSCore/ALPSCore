/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                              Matthias Troyer <troyer@comp-phys.org>             *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "ising.hpp"

bool stop_callback(boost::posix_time::ptime const & end_time) {
    static alps::mcsignal signal;
    return !signal.empty() || boost::posix_time::second_clock::local_time() > end_time;
}

template<typename T> typename alps::results_type<T>::type run_simulation(T & simulation, std::string const & checkpoint_path, std::size_t time_limit, bool resume) {
    if (resume)
        simulation.load(checkpoint_path);
    simulation.run(boost::bind(&stop_callback, boost::posix_time::second_clock::local_time() + boost::posix_time::seconds(time_limit)));
    simulation.save(checkpoint_path);
    return collect_results(simulation);
}

int main(int argc, char *argv[]) {
    alps::mcoptions options(argc, argv);
    if (options.valid) {
        alps::results_type<alps::mcbase>::type results;
        alps::parameters_type<alps::mcbase>::type parameters(options.input_file);
        std::string checkpoint_path = static_cast<std::string>(parameters.value_or_default("CHECKPOINT", "checkpoint"));
        if (options.type == alps::mcoptions::SINGLE) { 
            simulation_type simulation(parameters);
            results = run_simulation(
                  simulation
                , checkpoint_path
                , options.time_limit
                , options.resume
            );
        }
        #ifndef ALPS_NGS_SINGLE_THREAD
            else if (options.type == alps::mcoptions::THREADED) {
                alps::mcthreadedsim<simulation_type> simulation(parameters);
                results = run_simulation(
                      simulation
                    , checkpoint_path
                    , options.time_limit
                    , options.resume
                );
            }
        #endif
        #ifdef ALPS_HAVE_MPI
            else {
                boost::mpi::environment env(argc, argv);
                boost::mpi::communicator communicator;
                if (options.type == alps::mcoptions::MPI) {
                    alps::mcmpisim<simulation_type> simulation(parameters, communicator);
                    results = run_simulation(
                          simulation
                        , checkpoint_path + boost::lexical_cast<std::string>(communicator.rank())
                        , options.time_limit
                        , options.resume
                    );
                } else {
                    alps::mcmpisim<simulation_type> simulation(parameters, communicator);
                    results = run_simulation(
                          simulation
                        , checkpoint_path + boost::lexical_cast<std::string>(communicator.rank())
                        , options.time_limit
                        , options.resume
                    );
                }
            #endif
        }
        save_results(results, parameters, options.output_file, "/simulation/results/");
        for (alps::results_type<alps::mcbase>::type::const_iterator it = results.begin(); it != results.end(); ++it)
            std::cout << std::fixed << std::setprecision(5) << it->first << ": " << it->second->to_string() << std::endl;
        std::cout << "Sin of energy: " << sin(results["Energy"].get_mcdata<double>()) << std::endl
                  << "Mean of correlations: " << results["Correlations"].get_mcdata<std::vector<double> >().mean().size() << std::endl;
    } else {
        std::cerr << "Invalid options" << std::endl;
        std::abort();
    }
}
