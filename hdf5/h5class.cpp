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

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include <boost/filesystem.hpp>

#include <alps/hdf5.hpp>

class userdefined_class {
    public:
        userdefined_class(): a(1), b(10, 3) {}
        void serialize(alps::hdf5::iarchive & ar) { 
            ar 
                >> alps::make_pvp("a", a) 
                >> alps::make_pvp("b", b)
            ; 
        }
        void serialize(alps::hdf5::oarchive & ar) const { 
            ar 
                << alps::make_pvp("a", a) 
                << alps::make_pvp("b", b)
            ; 
        }
        void dump() {
            std::cout << "a: " << a << " b: (" << b.size() << "): [";
            for (std::size_t i = 0; i < b.size();  ++i)
                std::cout << b[i] << ( i < b.size() - 1 ? ", " : "");
            std::cout << "]" << std::endl;
        }
    private:
        int a;
        std::vector<long> b;
};
int main() {
    {
        alps::hdf5::oarchive oar("class.h5");
        {
            userdefined_class value;
            oar << alps::make_pvp("/class/scalar", value);
        }
        {
            std::vector<userdefined_class> value(5);
            oar << alps::make_pvp("/class/vector", value);
        }
    }
    {
        alps::hdf5::iarchive iar("class.h5");
        {
            userdefined_class value;
            iar >> alps::make_pvp("/class/scalar", value);
            value.dump();
        }
        {
            std::vector<userdefined_class> value;
            iar >> alps::make_pvp("/class/vector", value);
            std::cout << "vector: " << value.size() << std::endl;
        }
    }
    boost::filesystem::remove(boost::filesystem::path("class.h5"));
}
