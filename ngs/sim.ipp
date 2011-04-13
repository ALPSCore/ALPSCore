/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010 by Lukas Gamper <gamperl -at- gmail.com>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

bool stop_callback(boost::posix_time::ptime const & start_time, int time_limit) {
    static alps::mcsignal signal;
    return !signal.empty() || (time_limit > 0 && boost::posix_time::second_clock::local_time() > start_time + boost::posix_time::seconds(time_limit));
}

int main(int argc, char *argv[]) {
    alps::mcoptions options(argc, argv);
    if (options.valid && options.type == alps::mcoptions::SINGLE) {
        alps::parameters_type<simulation_type>::type params(options.input_file);
        simulation_type s(params);
        if (options.resume)
            s.load(static_cast<std::string>(params.value_or_default("DUMP", "dump")));
        s.run(boost::bind(&stop_callback, boost::posix_time::second_clock::local_time(), options.time_limit));
        s.save(static_cast<std::string>(params.value_or_default("DUMP", "dump")));
        alps::results_type<alps::mcmpisim<simulation_type> >::type results = collect_results(s);
        std::cout << results;
        {
            using namespace alps;
            std::cout << "Mean of Energy:       " << short_print(results["Energy"].mean<double>()) << std::endl;
            std::cout << "Mean of Correlations: " << short_print(results["Correlations"].mean<std::vector<double> >()) << std::endl;
            std::cout << "Sin of Energy:        " << short_print(sin(results["Energy"])) << std::endl;
            std::cout << "2 * Energy / 13:      " << short_print(2. * results["Energy"] / 13.) << std::endl;
            std::cout << "Correlations:         " << short_print(results["Correlations"]) << std::endl;
            std::cout << "Sign:                 " << short_print(results["Sign"]) << std::endl;
            std::cout << "Correlations / Sign:  " << short_print(results["Correlations"] / results["Sign"]) << std::endl;
            std::cout << "Sign / Correlations:  " << short_print(results["Sign"] / results["Correlations"]) << std::endl;
            std::cout << "                      " << short_print(1. / results["Correlations"] * results["Sign"]) << std::endl;
            std::cout << "-2 * Energy:          " << short_print(-2. * results["Energy"]) << std::endl;
        }
        save_results(results, params, options.output_file, "/simulation/results");
    }
    #ifdef ALPS_HAVE_MPI
        else if(options.valid && options.type == alps::mcoptions::MPI) {
            boost::mpi::environment env(argc, argv);
            boost::mpi::communicator c;
            alps::parameters_type<simulation_type>::type params(options.input_file);
            alps::mcmpisim<simulation_type> s(params, c);
            if (options.resume)
                s.load(static_cast<std::string>(params.value_or_default("DUMP", "dump")) + boost::lexical_cast<std::string>(c.rank()));
            s.run(boost::bind(&stop_callback, boost::posix_time::second_clock::local_time(), options.time_limit));
            s.save(static_cast<std::string>(params.value_or_default("DUMP", "dump")) + boost::lexical_cast<std::string>(c.rank()));
            alps::results_type<alps::mcmpisim<simulation_type> >::type results = collect_results(s);
            if (!c.rank()) {
                save_results(results, params, options.output_file, "/simulation/results");
                std::cout << results;
            }
        }
    #endif
}
