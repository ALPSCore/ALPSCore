/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
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

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/vector.hpp>

#include <boost/filesystem.hpp>

#include <iostream>
using namespace std;

int main()
{
    if (boost::filesystem::exists(boost::filesystem::path("test_hdf5_memory.h5")))
        boost::filesystem::remove(boost::filesystem::path("test_hdf5_memory.h5"));
    {
        alps::hdf5::archive oa("test_hdf5_memory.h5", "w");
        std::vector<std::complex<double> > foo(3);
        std::vector<double> foo2(3);
        oa << alps::make_pvp("/foo", foo);
        oa << alps::make_pvp("/foo2", foo2);
    }
    
    {
 
        std::vector<double> foo, foo2;
        try {
			alps::hdf5::archive ia("test_hdf5_memory.h5");
            ia >> alps::make_pvp("/foo", foo);
            ia >> alps::make_pvp("/foo2", foo2);
        } catch (exception e) {
            cout << "Exception caught: no complex value" << endl;
            boost::filesystem::remove(boost::filesystem::path("test_hdf5_memory.h5"));
            return EXIT_SUCCESS;
        }
    }
    boost::filesystem::remove(boost::filesystem::path("test_hdf5_memory.h5"));
    return EXIT_SUCCESS;
}