/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "alps/params.hpp"
#include "gtest/gtest.h"
#include "alps/testing/unique_file.hpp"

#include "boost/lexical_cast.hpp"

#include <cstdio>
#include <fstream>

// Repeated parameters in the INI file
TEST(param,RepeatingInFile) {
    //create a file name
  std::string pfilename(alps::testing::temporary_filename("pfile.ini."));

   // Generate INI file
   {
     std::ofstream pfile(pfilename.c_str());
     pfile <<
         "parname = 1\n"
         "parname = 2\n";
   }

   // Imitate the command line args
   const char* argv[]={"THIS_PROGRAM", pfilename.c_str()};
   const int argc=sizeof(argv)/sizeof(*argv);

   //define the parameters
   alps::params p(argc,argv);
   p.description("This is a test program").
       define<int>("parname","repeated parameter");

   EXPECT_THROW(p["parname"],boost::program_options::multiple_occurrences);
}

// Repeating parameters in the command line
TEST(param,RepeatingInCmdline) {
    // Imitate the command line args
    const char* argv[]={"THIS_PROGRAM","--parname=1","--parname=2"};
    const int argc=sizeof(argv)/sizeof(*argv);

    //define the parameters
    alps::params p(argc,argv);
    p.description("This is a test program").
        define<int>("parname","repeated parameter");

    EXPECT_THROW(p["parname"],boost::program_options::multiple_occurrences);
}

// Command-line options overriding file options
TEST(param,CmdlineOverride)
{
    //create a file name
    std::string pfilename(alps::testing::temporary_filename("pfile.ini."));
    
    // Generate INI file
    {
        std::ofstream pfile(pfilename.c_str());
        pfile <<
            "param1 = 111\n"
            "param2 = 222\n";
    }

    // Imitate the command line args
    const char* argv[]={"THIS_PROGRAM",         // argv[0]
                        pfilename.c_str(),      // filename is the 1st argument
                        "--param1=999",         // override param1
                        "--param3=333",         // one more parameter
                        "--trigger_opt" };      // a trigger option  
    const int argc=sizeof(argv)/sizeof(*argv);
    
    alps::params p(argc, argv);
    p.
        define<int>("param1","Parameter 1").
        define<int>("param2","Parameter 2").
        define<int>("param3","Parameter 3").
        define("trigger_opt","Trigger param");

    EXPECT_EQ(999,p["param1"]);
    EXPECT_EQ(222,p["param2"]);
    EXPECT_EQ(333,p["param3"]);
    EXPECT_TRUE(bool(p["trigger_opt"]));
}

