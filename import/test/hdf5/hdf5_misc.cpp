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

#include <alps/hdf5.hpp>

#include <boost/filesystem.hpp>

class my_class {
    public:
        my_class(double v = 0): d(v) {}
        void save(alps::hdf5::archive & ar) const {
            using alps::make_pvp;
            ar << make_pvp("value", d);
        }
        void load(alps::hdf5::archive & ar) { 
            using alps::make_pvp;
            ar >> make_pvp("value", d); 
        }
    private:
        double d;
};

int main () {

    if (boost::filesystem::exists(boost::filesystem::path("data.h5")))
        boost::filesystem::remove(boost::filesystem::path("data.h5"));

    {
        alps::hdf5::archive ar("data.h5", "w");
        ar << alps::make_pvp("/value", 42);
    }
    
    {
        alps::hdf5::archive ar("data.h5");
        int i;
        ar >> alps::make_pvp("/value", i);
    }

    {
        alps::hdf5::archive ar("data.h5");
        std::string s;
        ar >> alps::make_pvp("/value", s);
    }

    {
        alps::hdf5::archive ar("data.h5", "w");
        std::vector<double> vec(5, 42);
        ar << alps::make_pvp("/path/2/vec", vec);
    }
    
    {
        std::vector<double> vec;
        // fill the vector
        alps::hdf5::archive ar("data.h5");
        ar >> alps::make_pvp("/path/2/vec", vec);
    }

    {
        std::string str("foobar");
        alps::hdf5::archive ar("data.h5", "w");
        ar << alps::make_pvp("/foo/bar", str);
    }
    
    {
        alps::hdf5::archive ar("data.h5");
        std::string str;
        ar >> alps::make_pvp("/foo/bar", str);
    }

    {
        long *d = new long[17];
        // fill the array
        alps::hdf5::archive ar("data.h5", "w");
        ar << alps::make_pvp("/c/array", d, 17);
        delete[] d;
    }

    {
        alps::hdf5::archive ar("data.h5");
        std::size_t size = ar.extent("/c/array")[0];
        long *d = new long[size];
        ar >> alps::make_pvp("/c/array", d, size);
        delete[] d;
    }

    {
        {
                my_class c(42);
                alps::hdf5::archive ar("data.h5", "w");
                ar << alps::make_pvp("/my/class", c);
        }
        {
                my_class c;
                alps::hdf5::archive ar("data.h5");
                ar >> alps::make_pvp("/my/class", c);
        }
    }

    {
        alps::hdf5::archive ar("data.h5", "w"); 
        // the parent of an attribute must exist
        ar.create_group("/foo");
        ar << alps::make_pvp("/foo/@bar", std::string("hello"));
    }

    {
        alps::hdf5::archive ar("data.h5");
        std::string str;
        ar >> alps::make_pvp("/foo/@bar", str);
    }

    boost::filesystem::remove(boost::filesystem::path("data.h5"));
    return 0;
}
