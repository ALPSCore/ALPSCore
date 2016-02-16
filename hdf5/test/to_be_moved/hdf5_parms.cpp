/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <boost/filesystem.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/parameter.h>

using namespace std;

int main ()
{
    
    if (boost::filesystem::exists("parms.h5") && boost::filesystem::is_regular_file("parms.h5"))
        boost::filesystem::remove("parms.h5");
    
    alps::Parameters p, p2;
    p["a"] = 10;
    p["b"] = "test";
    p["c"] = 10.;
    p["d"] = 5.;
    
    {
        alps::hdf5::archive ar("parms.h5", "a");
        ar << alps::make_pvp("/parameters", p);
    }
    {
        alps::hdf5::archive ar("parms.h5", "r");
        alps::Parameters pin;
        ar >> alps::make_pvp("/parameters", pin);
        cout << "Reading 1:" << endl << pin;
    }
    
    // "a" is modified from int to double
    // "c" is modified from double to double (but with decimals)
    p2["a"] = 10.5;
    p2["c"] = 5.2;
    {
        alps::hdf5::archive ar("parms.h5", "a");
        ar << alps::make_pvp("/parameters", p2);
    }
    {
        alps::hdf5::archive ar("parms.h5", "r");
        alps::Parameters pin;
        ar >> alps::make_pvp("/parameters", pin);
        cout << "Reading 2:" << endl << pin;
    }
    
    // "d" is modified from double to string
    p2["d"] = "newtype";
    {
        alps::hdf5::archive ar("parms.h5", "a");
        ar << alps::make_pvp("/parameters", p2);
    }
    {
        alps::hdf5::archive ar("parms.h5", "r");
        alps::Parameters pin;
        ar >> alps::make_pvp("/parameters", pin);
        cout << "Reading 3:" << endl << pin;
    }
    
    return 0;
}
