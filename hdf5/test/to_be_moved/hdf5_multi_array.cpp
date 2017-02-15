/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
