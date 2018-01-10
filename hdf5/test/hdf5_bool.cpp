/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/pair.hpp>
#include <iostream>
#include "gtest/gtest.h"

using namespace std;
using namespace alps;

namespace alps { namespace hdf5 { class archive; } }

enum E { E1, E2 };
void save(
      alps::hdf5::archive & ar
    , std::string const & path
    , E const & value
    , std::vector<std::size_t> /*size*/ = std::vector<std::size_t>()
    , std::vector<std::size_t> /*chunk*/ = std::vector<std::size_t>()
    , std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()
)
{
    if( value == E1 )     ar << alps::make_pvp(path, "E1");
    else if( value == E2 )     ar << alps::make_pvp(path, "E2");
}
void load(
      alps::hdf5::archive & ar
    , std::string const & path
    , E & value
    , std::vector<std::size_t> /*chunk*/ = std::vector<std::size_t>()
    , std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()
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

TEST(hdf5, TestingIoOfBoolVars){
    A a;
    a.b.b = true; a.b.p = std::make_pair(3,4); a.b.e = E1;
    a.c.b = false; a.c.u = 1;
    {
        hdf5::archive ar("test_hdf5_bool.h5","w");
        ar << make_pvp("/true",true);
        ar << make_pvp("/false",false);
        ar << make_pvp("/a",a);
    }
    {
        hdf5::archive ar("test_hdf5_bool.h5", "r");
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
}
