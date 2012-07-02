/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/params.hpp>
#include <alps/ngs/signal.hpp>

#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char *argv[]) {

    alps::ngs::signal::listen();

    // load a hdf5 file into a parameter object
    alps::hdf5::archive ar("param.h5");
    alps::params params_h5(ar);

    for (alps::params::const_iterator it = params_h5.begin(); it != params_h5.end(); ++it)
        std::cout << it->first << ":" << it->second << std::endl;

    std::cout << std::endl;

    // save params to file
    {
        alps::hdf5::archive ar("param.out.h5", "w");
        ar << make_pvp("/parameters", params_h5);
    }

    // load a text file into a parameter object
    alps::params params_txt(boost::filesystem::path("param.txt"));

    for (alps::params::const_iterator it = params_txt.begin(); it != params_txt.end(); ++it)
        std::cout << it->first << ":" << it->second << std::endl;

    std::cout << std::endl;
    params_txt["dbl"] = 1e-8;
    std::cout << params_txt["dbl"] << " " << static_cast<double>(params_txt["dbl"]) << std::endl;

    std::cout << std::endl;
    params_txt["NOT_EXISTING_PARAM"] = "";
    int i = params_txt["NOT_EXISTING_PARAM"] | 2048;
    int j(params_txt["NOT_EXISTING_PARAM"] | 2048);
    std::cout << static_cast<int>(params_txt["NOT_EXISTING_PARAM"] | 2048) << " " << i << " " << j << std::endl;
}
