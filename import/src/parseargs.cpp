/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2013 by Lukas Gamper <gamperl@gmail.com>                   *
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

#include <alps/parseargs.hpp>

#include <boost/program_options.hpp>

#include <sstream>
#include <iostream>

namespace alps {

    parseargs::parseargs(int argc, char *argv[]) {
        boost::program_options::options_description options("Options");
        options.add_options()
            ("continue,c", "load simulation from checkpoint")
            ("timelimit,T", boost::program_options::value<std::size_t>(&timelimit)->default_value(0), "time limit for the simulation")
            ("Tmin,i", boost::program_options::value<std::size_t>(&tmin)->default_value(1), "minimum time to check if simulation has finished")
            ("Tmax,a", boost::program_options::value<std::size_t>(&tmax)->default_value(600), "maximum time to check if simulation has finished")
            ("inputfile", boost::program_options::value<std::string>(&input_file), "input file in hdf5 or xml format")
            ("outputfile", boost::program_options::value<std::string>(&output_file)->default_value(""), "output file in hdf5 format")
        ;
        boost::program_options::positional_options_description positional;
        positional
            .add("inputfile", 1)
            .add("outputfile", 1)
        ;

        try {
            boost::program_options::variables_map variables;
            boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(options).positional(positional).run(), variables);
            boost::program_options::notify(variables);

            resume = variables.count("continue");
            if (output_file.empty())
                output_file = input_file.substr(0, input_file.find_last_of('.')) +  ".out.h5";
        } catch (...) {
    		std::stringstream ss;
            ss << "usage: [-T timelimit] [-i tmin] [-a tmax] [-c] inputfile [outputfile]" << std::endl
               << options << std::endl;
            std::cerr << ss.str();
            std::abort();
        }
    }

}
