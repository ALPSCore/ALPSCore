/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file hdf5_open.cpp test basic archive object ctor and file opening */

#include <alps/hdf5/archive.hpp>
#include <alps/testing/unique_file.hpp>
#include <gtest/gtest.h>
#include <cstdio>

/// Test for "a","w","r" open modes
TEST(hdf5,OpenModesAWR) {
    alps::testing::unique_file ufile("hdf5_open.h5.", alps::testing::unique_file::REMOVE_NOW);
    const std::string& h5name=ufile.name();
    {
        ASSERT_THROW(alps::hdf5::archive ar(h5name,"r"),  alps::hdf5::archive_not_found);
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
        EXPECT_THROW(ar["/int4"] << 4, alps::hdf5::archive_error);
    }
    {
        alps::hdf5::archive ar(h5name,"");
        int n=0;
        ar["/int1"] >> n;
        EXPECT_EQ(1,n);
        ar["/int2"] >> n;
        EXPECT_EQ(2,n);
        ar["/int3"] >> n;
        EXPECT_EQ(3,n);
        EXPECT_THROW(ar["/int4"] << 4, alps::hdf5::archive_error);
    }
    {
        alps::hdf5::archive ar(h5name);
        int n=0;
        ar["/int1"] >> n;
        EXPECT_EQ(1,n);
        ar["/int2"] >> n;
        EXPECT_EQ(2,n);
        ar["/int3"] >> n;
        EXPECT_EQ(3,n);
        EXPECT_THROW(ar["/int4"] << 4, alps::hdf5::archive_error);
    }
}

/// Test for incorrect open modes
TEST(hdf5,OpenModesWrong) {
    alps::testing::unique_file ufile("hdf5_open.h5.", alps::testing::unique_file::REMOVE_NOW);
    const std::string& h5name=ufile.name();
    {
      alps::hdf5::archive ar(h5name,"w");
      ar["int"]=1;
    }
    {
      EXPECT_THROW(alps::hdf5::archive ar(h5name,"wl"), alps::hdf5::wrong_mode);
    }
    {
      EXPECT_THROW(alps::hdf5::archive ar(h5name,"no_such_mode"), alps::hdf5::wrong_mode);
    }
}
