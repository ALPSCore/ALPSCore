/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/hdf5/archive.hpp>
#include <alps/testing/unique_file.hpp>

#include <vector>

#include "gtest/gtest.h"

class TestHDF5Attributes : public ::testing::Test {
  public:
    std::string fname_;

    TestHDF5Attributes() {
        fname_=alps::testing::temporary_filename("attr.h5.");
    }

    ~TestHDF5Attributes() {
        if (!fname_.empty()) remove(fname_.c_str());
    }
};

namespace ah5=alps::hdf5;

TEST_F(TestHDF5Attributes, Data) {
    {
        ah5::archive ar(fname_,"w");
        ar["/int"] << int(123);
        ar["/int@attr"]="My data attribute";
    }
    ah5::archive ar(fname_,"r");
    ASSERT_TRUE(ar.is_attribute("/int@attr"));
    std::string attr;
    ar["/int@attr"] >> attr;
    EXPECT_EQ("My data attribute", attr);
}

TEST_F(TestHDF5Attributes, Group) {
    {
        ah5::archive ar(fname_,"w");
        ar["/group/int"]=int(123);
        ar["/group@attr"]="My group attribute";
    }
    ah5::archive ar(fname_,"r");
    ASSERT_TRUE(ar.is_attribute("/group@attr"));
    std::string attr;
    ar["/group@attr"] >> attr;
    EXPECT_EQ("My group attribute", attr);
}

TEST_F(TestHDF5Attributes, Root) {
    {
        ah5::archive ar(fname_,"w");
        ar["/group/int"]=int(123);
        ar["/@attr"]="My root attribute";
    }
    ah5::archive ar(fname_,"r");
    ASSERT_TRUE(ar.is_attribute("/@attr"));
    std::string attr;
    ar["/@attr"] >> attr;
    EXPECT_EQ("My root attribute", attr);
}

TEST_F(TestHDF5Attributes, Current) {
    {
        ah5::archive ar(fname_,"w");
        ar["/group/int2"]=int(321);
        ar.set_context("/group");
        ar["@attr"]="My group context attribute";
    }
    ah5::archive ar(fname_,"r");
    ar.set_context("/group");
    ASSERT_TRUE(ar.is_attribute("@attr"));
    std::string attr;
    ar["@attr"] >> attr;
    EXPECT_EQ("My group context attribute", attr);
}
