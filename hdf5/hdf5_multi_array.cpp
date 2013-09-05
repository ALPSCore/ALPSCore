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
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/multi_array.hpp>

#include <boost/multi_array.hpp>
#include <boost/filesystem.hpp>

#include <vector>

using namespace std;
using boost::multi_array;

int main()
{
    if (boost::filesystem::exists(boost::filesystem::path("test_hdf5_multi_array.h5")))
        boost::filesystem::remove(boost::filesystem::path("test_hdf5_multi_array.h5"));

    multi_array<double,2> a( boost::extents[3][3] );
    multi_array<double,2> b( boost::extents[4][4] );

    // Write
    {
        alps::hdf5::archive ar("test_hdf5_multi_array.h5","a");
        vector< multi_array<double,2> > v(2,a);
        ar << alps::make_pvp("uniform",v);
        v.push_back(b);
        ar << alps::make_pvp("nonuniform",v);
    }

    // Read
    {
        alps::hdf5::archive ar("test_hdf5_multi_array.h5","r");
        vector< multi_array<double,2> > w;
        ar >> alps::make_pvp("nonuniform",w);
        cout << "read nonuniform" << endl;
        ar >> alps::make_pvp("uniform",w); // throws runtime_error
        cout << "read uniform" << endl;
    }
    
    if (boost::filesystem::exists(boost::filesystem::path("test_hdf5_multi_array.h5")))
        boost::filesystem::remove(boost::filesystem::path("test_hdf5_multi_array.h5"));
    return 0;
}
