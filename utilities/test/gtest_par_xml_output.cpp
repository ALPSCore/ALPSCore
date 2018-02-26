/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include "alps/utilities/gtest_par_xml_output.hpp"
#include "gtest/gtest.h"

TEST(gtest_par_xml_output, main)
{
    // Sample command line initializer
    const std::string argv_init[]={
        std::string("program_name"),  // 0
        std::string("--some_option"), // 1
        std::string("--gtest_other=abc"), // 2
        std::string("--gtest_output=xmlsome-other-type"), // 3
        std::string("--gtest_output=xml"), // 4
        std::string("--gtest_output=xml:/some/dirname/"), // 5
        std::string("--gtest_output=xml:/some/filename"), // 6
        std::string("--gtest_output=xml:/some/filename.ext"), // 7
        std::string("--gtest_output=xml:/some/filename."), // 8
        std::string("--some_more_options") // 9
    };
    const int argc=sizeof(argv_init)/sizeof(*argv_init);

    // Copy of the command line
    std::vector<std::string> argv_copy(argc);
    std::copy(argv_init, argv_init+argc, argv_copy.begin());
    
    // Command line to be scanned
    char** argv=new char*[argc];
    for (int i=0; i<argc; ++i) {
        argv[i]=const_cast<char*>(argv_copy[i].c_str()); // dirty, but would work here
    }

    alps::gtest_par_xml_output tweak; // hold memory
    tweak(123, argc, argv); // do tweaking

    {
        // These arguments should stay intact
        int idx[]={0, 1, 2, 3, 9};
        for (unsigned int i=0; i<sizeof(idx)/sizeof(*idx); ++i) {
            int ii=idx[i];
            EXPECT_EQ(argv_init[ii], argv[ii]) << "Unexpected change in argv[" << ii << "]";
        }
    }
    
    // These arguments should change:
    EXPECT_EQ("--gtest_output=xml:test_details123.xml",   std::string(argv[4])) << "Wrong \"=xml\"";
    EXPECT_EQ("--gtest_output=xml:/some/dirname123/",     std::string(argv[5])) << "wrong xml=dir/";
    EXPECT_EQ("--gtest_output=xml:/some/filename123",     std::string(argv[6])) << "wrong xml=file";
    EXPECT_EQ("--gtest_output=xml:/some/filename123.ext", std::string(argv[7])) << "wrong xml=file.ext";
    EXPECT_EQ("--gtest_output=xml:/some/filename123.", std::string(argv[8])) << "wrong xml=file.";

    delete[] argv;
}
