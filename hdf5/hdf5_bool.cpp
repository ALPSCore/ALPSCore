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
#include <alps/hdf5/pair.hpp>
#include <iostream>

using namespace std;
using namespace alps;

namespace alps { namespace hdf5 { class archive; } }

enum E { E1, E2 };
void save(
      alps::hdf5::archive & ar
    , std::string const & path
    , E const & value
    , std::vector<std::size_t> size = std::vector<std::size_t>()
    , std::vector<std::size_t> chunk = std::vector<std::size_t>()
    , std::vector<std::size_t> offset = std::vector<std::size_t>()
)
{
    if( value == E1 )     ar << alps::make_pvp(path, "E1");
    else if( value == E2 )     ar << alps::make_pvp(path, "E2");
}
void load(
      alps::hdf5::archive & ar
    , std::string const & path
    , E & value
    , std::vector<std::size_t> chunk = std::vector<std::size_t>()
    , std::vector<std::size_t> offset = std::vector<std::size_t>()
)
{
    std::string s;
    ar >> alps::make_pvp(path, s);
    if     ( s == "E1" ) value = E1;
    else if( s == "E2"  ) value = E2;
    else                 throw std::runtime_error("invalid line type "+s);
}

class A
{
public:
        void load(alps::hdf5::archive& ar) { ar >> make_pvp("b",b) >> make_pvp("c",c); }
        void save(alps::hdf5::archive& ar) const { ar << make_pvp("b",b) << make_pvp("c",c); }

    struct B { 
        bool b; std::pair<unsigned,unsigned> p; E e; 
        void load(alps::hdf5::archive& ar) { ar >> make_pvp("b",b) >> make_pvp("p",p) >> make_pvp("e",e); }
        void save(alps::hdf5::archive& ar) const { ar << make_pvp("b",b) << make_pvp("p",p) << make_pvp("e",e); }
    };
    struct C { 
        bool b; unsigned u;
        void load(alps::hdf5::archive& ar) { ar >> make_pvp("b",b) >> make_pvp("u",u); };
        void save(alps::hdf5::archive& ar) const { ar << make_pvp("b",b) << make_pvp("u",u); };
    };

    B b;
    C c;
};

int main()
{
    A a;
    a.b.b = true; a.b.p = std::make_pair(3,4); a.b.e = E1;
    a.c.b = false; a.c.u = 1;
    {
        hdf5::archive ar("test_hdf5_bool.h5",1);
        ar << make_pvp("/true",true);
        ar << make_pvp("/false",false);
        ar << make_pvp("/a",a);
    }
    {
        hdf5::archive ar("test_hdf5_bool.h5", 0);
        bool bb, bc, bt, bf;
        ar 
            >> make_pvp("/a/b/b",bb) 
            >> make_pvp("/a/c/b",bc) 
            >> make_pvp("/true",bt) 
            >> make_pvp("/false",bf)
        ;
        std::cout << "Read bb=" << bb << ", should be " << a.b.b << endl;
        std::cout << "Read bc=" << bc << ", should be " << a.c.b << endl;
        std::cout << "Read bt=" << bt << ", should be " << true << endl;
        std:: cout << "Read bf=" << bf << ", should be " << false << endl;
    }
    return 0;
}
