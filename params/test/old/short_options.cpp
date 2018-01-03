/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <fstream>
#include "alps/params.hpp"
#include "alps/testing/unique_file.hpp"
#include "gtest/gtest.h"

//Dummy function to imitate use of a variable to supress spurious compiler warnings
static inline void dummy_use(const void*) {}

// Test interaction with boost::program_options --- short options are prohibited
TEST(param, ProgramOptions)
{
    const char* argv[]={ "THIS PROGRAM", "--param=123" };
    const int argc=sizeof(argv)/sizeof(*argv);

    alps::params p(argc, argv);

    EXPECT_THROW((p.define<int>("param,p", "Int parameter")), alps::params::invalid_name);

    // if allowed boost::po style of option names as "long,short", these 2 below would crash on assertion failure.
    { EXPECT_THROW({int i=p["param"]; dummy_use(&i);}, alps::params::uninitialized_value); }
    { EXPECT_THROW({int i=p["p"]; dummy_use(&i);}, alps::params::uninitialized_value); } 
}

// Shortened versions of options in the command line.
TEST(param, ShortenedInCmdline)
{
    const char* argv[]={ "THIS PROGRAM", "--par=123" };
    const int argc=sizeof(argv)/sizeof(*argv);

    alps::params p(argc, argv);
    p.define<int>("param", "Int parameter");

    EXPECT_EQ(123,int(p["param"]));
    EXPECT_THROW({int i=p["par"]; dummy_use(&i);}, alps::params::uninitialized_value);
}

// Shortened versions of options in the INI file -- not allowed
TEST(param, ShortenedInFile)
{
   std::string pfilename(alps::testing::temporary_filename("pfile.ini."));
    const char* argv[]={ "THIS PROGRAM", pfilename.c_str() };
    const int argc=sizeof(argv)/sizeof(*argv);

    // Generate INI file
    {
        std::ofstream pfile(pfilename.c_str());
        pfile << "par = 123\n";
    }
    
    alps::params p(argc, argv);
    p.define<int>("param", "Int parameter");

    EXPECT_THROW({int i=p["param"]; dummy_use(&i);}, alps::params::uninitialized_value);
    EXPECT_THROW({int i=p["par"]; dummy_use(&i);}, alps::params::uninitialized_value);
}

// Shorter and longer options in the command line.
TEST(param, ShortAndLongCmdline)
{
    const char* argv[]={ "THIS PROGRAM", "--par=123", "--param=456" };
    const int argc=sizeof(argv)/sizeof(*argv);

    alps::params p(argc, argv);
    p.define<int>("param", "Int parameter");

    // Parsing occurs here, and program_options throwns an exception:
    EXPECT_THROW({int i=p["par"]; dummy_use(&i);}, boost::program_options::multiple_occurrences);
    // Parsing occurs here again, and program_options throwns an exception again:
    EXPECT_THROW({int i=p["param"]; dummy_use(&i);}, boost::program_options::multiple_occurrences);
}

// Shorter and longer options defined
TEST(param, ShortAndLongDefined)
{
    const char* argv[]={ "THIS PROGRAM", "--par=123", "--param=456" };
    const int argc=sizeof(argv)/sizeof(*argv);

    alps::params p(argc, argv);
    p   .define<int>("param", "Int parameter 1")
        .define<int>("par", "Int parameter 2");

    EXPECT_EQ(123,int(p["par"]));
    EXPECT_EQ(456,int(p["param"]));
}

// Shorter and longer options in the INI file --- are distinct
TEST(param, ShortAndLongFile)
{
    std::string pfilename(alps::testing::temporary_filename("pfile.ini."));
    const char* argv[]={ "THIS PROGRAM", pfilename.c_str() };
    const int argc=sizeof(argv)/sizeof(*argv);

    // Generate INI file
    {
        std::ofstream pfile(pfilename.c_str());
        pfile << "par = 123\n"
              << "param = 456\n";
    }
    
    alps::params p(argc, argv);
    p.define<int>("param", "Int parameter");

    EXPECT_THROW({int i=p["par"]; dummy_use(&i);}, alps::params::uninitialized_value);
    EXPECT_EQ(456, int(p["param"]));
}


