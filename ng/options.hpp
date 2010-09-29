/*****************************************************************************
 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations
 *
 * ALPS Libraries
 *
 * Copyright (C) 2010 by Lukas Gamper <gamperl@gmail.com>
 *                       Matthias Troyer <troyer@comp-phys.org>
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

#include <boost/program_options.hpp>

#ifndef ALPS_NG_OPTIONS_HPP
#define ALPS_NG_OPTIONS_HPP

namespace alps {
    namespace ng {
        class options {
            public:
                typedef enum { SINGLE, MPI } execution_types;

                options(int argc, char* argv[]) : valid(false), reload(false), type(SINGLE) {
                    boost::program_options::options_description desc("Allowed options");
                    desc.add_options()
                        ("help", "produce help message")
                        ("single", "run single process")
                        ("mpi", "run in parallel using MPI")
                        ("reload", "load simulation from checkpoint")
                        ("time-limit,T", boost::program_options::value<std::size_t>(&time_limit)->default_value(0), "time limit for the simulation")
                        ("input-file", boost::program_options::value<std::string>(&input_file), "input file in hdf5 format")
                        ("output-file", boost::program_options::value<std::string>(&output_file)->default_value("sim.h5"), "output file in hdf5 format");
                    boost::program_options::positional_options_description p;
                    p.add("input-file", 1);
                    p.add("output-file", 2);
                    boost::program_options::variables_map vm;
                    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
                    boost::program_options::notify(vm);
                    if (!(valid = !vm.count("help")))
                        std::cout << desc << std::endl;
                    else if (input_file.empty())
                        throw std::invalid_argument("No job file specified");
                    if (vm.count("mpi"))
                        type = MPI;
                    if (vm.count("reload")) // CHANGE: continue
                        reload = true;
                }

                bool valid;
                bool reload;
                std::size_t time_limit;
                std::string input_file;
                std::string output_file;
                execution_types type;
        };
    }
}

#endif
