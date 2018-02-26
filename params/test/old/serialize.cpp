/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <sstream>

#include "alps/testing/unique_file.hpp"
#include <alps/hdf5/archive.hpp>

#include "alps/params.hpp"
#include "gtest/gtest.h"

// FIXME: rewrite the test using convenience test classes from "param_generators.hpp"
//        and test ALL supported types.

// FIXME: add test for saving the options as individual H5 datafields.

TEST(param, Archive)
{
    typedef std::vector<double> dblvec_type;
  
    // Prepare parameters
    const char* argv[]={ "", "--param1=111", "--flag" };
    const int argc=sizeof(argv)/sizeof(*argv);
    alps::params p(argc,argv);

    p.description("Archiving test").
        define("flag","trigger param").
        define<int>("param1","integer 1").
        define<double>("param2",22.25,"double").
        define<dblvec_type>("vparam","vector of doubles");
    
    p["param3"]=333;
    p["nosuchparam"]; // FIXME: test that it did not spring into existance.
    dblvec_type dv(3);
    dv[0]=1.25;
    dv[1]=2.50;
    dv[2]=3.75;
    p["vparam"]=dv;

    EXPECT_FALSE(p.is_restored());
    EXPECT_THROW(p.get_archive_name(), alps::params::not_restored);


    // Save to archive
    std::string filename(alps::testing::temporary_filename("hdf5_file.h5."));
    {
        alps::hdf5::archive oar(filename, "w");
        p.save(oar);
    }

    // Load from archive
    alps::params p2;
    {
        alps::hdf5::archive iar(filename, "r");
        p2.load(iar);
    }
    EXPECT_FALSE(p2["help"].as<bool>());
    EXPECT_TRUE(p2["flag"].as<bool>());
    EXPECT_EQ(111, p2["param1"]);
    EXPECT_EQ(22.25, p2["param2"]);
    EXPECT_EQ(333, p2["param3"]);
    EXPECT_EQ(dv.size(), p2["vparam"].as<dblvec_type>().size());
    EXPECT_EQ(dv, p2["vparam"].as<dblvec_type>());

    EXPECT_FALSE(p2.is_restored());
    EXPECT_THROW(p2.get_archive_name(), alps::params::not_restored);

    // Create from archive
    alps::hdf5::archive iar3(filename, "r");
    alps::params p3(iar3, "/");
    EXPECT_EQ(111, p3["param1"]);
    EXPECT_EQ(22.25, p3["param2"]);
    EXPECT_EQ(333, p3["param3"]);
    EXPECT_EQ(dv.size(), p2["vparam"].as<dblvec_type>().size());
    EXPECT_EQ(dv, p2["vparam"].as<dblvec_type>());

    EXPECT_FALSE(p3.is_restored());
    EXPECT_THROW(p3.get_archive_name(), alps::params::not_restored);
    

    // Create from commandline containing an archive
    const char* argv2[]={ "", filename.c_str(), "--param1=999" };
    const int argc2=sizeof(argv2)/sizeof(*argv2);
    alps::params p4(argc2,argv2,"/");
    EXPECT_EQ(999, p4["param1"]); // note: not 111; cmdline is re-processed!
    EXPECT_EQ(22.25, p4["param2"]);
    EXPECT_EQ(333, p4["param3"]);
    EXPECT_EQ(dv.size(), p2["vparam"].as<dblvec_type>().size());
    EXPECT_EQ(dv, p2["vparam"].as<dblvec_type>());

    EXPECT_TRUE(p4.is_restored());
    EXPECT_EQ(filename, p4.get_archive_name());
}

