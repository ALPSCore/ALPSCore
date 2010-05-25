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

#include <alps/hdf5.hpp>

#include <vector>

#include <boost/filesystem.hpp>

typedef enum { PLUS, MINUS } enum1_type;
inline alps::hdf5::oarchive & serialize(alps::hdf5::oarchive & ar, std::string const & p, enum1_type const & v) {
    switch (v) {
        case PLUS: ar << alps::make_pvp(p, std::string("plus"));
        case MINUS: ar << alps::make_pvp(p, std::string("minus"));
    }
    return ar;
}
inline alps::hdf5::iarchive & serialize(alps::hdf5::iarchive & ar, std::string const & p, enum1_type & v) {
    std::string s;
    ar >> alps::make_pvp(p, s);
    v = (s == "plus" ? PLUS : MINUS);
    return ar;
}
template<typename T> void test_enum(T & v, std::vector<T> w, T c[2]) {
    {
        alps::hdf5::oarchive oar("enum.h5");
        oar << alps::make_pvp("/enum/scalar", v);
        oar << alps::make_pvp("/enum/vector", w);
        oar << alps::make_pvp("/enum/c_arr", c, 2);
    }
    {
        alps::hdf5::iarchive iar("enum.h5");
        iar >> alps::make_pvp("/enum/scalar", v);
        std::cout << v << std::endl;
    }
}
int main() {
    {
        enum1_type v = PLUS;
        std::vector<enum1_type> w(2, MINUS);
        enum1_type c[2] = { PLUS, MINUS };
        test_enum(v, w, c);
    }
    boost::filesystem::remove(boost::filesystem::path("enum.h5"));
}
