/** @file hdf5_open.cpp test basic archive object ctor and file opening */

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/temporary_filename.hpp>
#include <gtest/gtest.h>
#include <cstdio>

/// Test for "a","w","r" open modes
TEST(hdf5,OpenModesAWR) {
    std::string h5name=alps::temporary_filename("hdf5_open")+".h5";
    {
        ASSERT_THROW(alps::hdf5::archive ar(h5name,"r"),alps::hdf5::archive_not_found);
        ASSERT_THROW(alps::hdf5::archive ar2(h5name,"r"), alps::hdf5::archive_not_found);
    }
    {
        alps::hdf5::archive ar(h5name,"w"); // same as "a"
        ar["/int1"]=1;
    }
    {
        alps::hdf5::archive ar(h5name,"w"); // same as "a"
        ar["/int2"]=2;
    }
    {
        alps::hdf5::archive ar(h5name,"a");
        ar["/int3"]=3;
    }
    {
        alps::hdf5::archive ar(h5name,"r");
        int n=0;
        ar["/int1"] >> n;
        EXPECT_EQ(1,n);
        ar["/int2"] >> n;
        EXPECT_EQ(2,n);
        ar["/int3"] >> n;
        EXPECT_EQ(3,n);
    }
    std::remove(h5name.c_str());
}
