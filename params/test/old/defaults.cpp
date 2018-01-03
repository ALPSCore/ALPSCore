/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "alps/params.hpp"
#include "gtest/gtest.h"
#include "alps/testing/unique_file.hpp"

#include "boost/lexical_cast.hpp"

#include <fstream>

// To imitatate using a variable to supress spurious warnings
static inline void dummy_use(const void*) {}

// Service function to construct a parameters object from a commandline:
// contains option @param @name1 with value @param val1 and
// option @param @name2 with value @param val2 .
template <typename T>
alps::params get_cmdline_param(const std::string& name1, const std::string& name2, T val1, T val2)
{
    std::string arg1= "--" + name1 + "=" + boost::lexical_cast<std::string>(val1);
    std::string arg2= "--" + name2 + "=" + boost::lexical_cast<std::string>(val2);
    const char* argv[] = { "this program", arg1.c_str(), arg2.c_str() };
    const int argc=sizeof(argv)/sizeof(*argv);
    return alps::params(argc,argv);
}
  
// Service function to construct a parameters object from a file:
// the file contains option @param @name1 with value @param val1 and
// option @param @name2 with value @param val2 .
template <typename T>
alps::params get_file_param(const std::string& name1, const std::string& name2, T val1, T val2)
{
    //create a file name
    std::string pfilename(alps::testing::temporary_filename("pfile.ini."));
    // Generate INI file
    {
        std::ofstream pfile(pfilename.c_str());
        pfile
            << name1 << " = " << val1 << "\n"
            << name2 << " = " << val2 << "\n";
    }
    const char* argv[] = { "this program", pfilename.c_str() };
    const int argc=sizeof(argv)/sizeof(*argv);
    return alps::params(argc,argv);
}

// Scalars and strings with default values;
// Tests the following parameters:
// 1) Existing parameter with value @param val1, defined with default @param defval2
// 2) Existing parameter with value @param val2, defined without a default
// 3) Non-existing parameter defined with default value @param defval2
// 4) Non-existing parameter defined without a default
// The parameter object is constructed from a file if @param from_file is True, from commandline otherwise.
template <typename T>
void TestDefaults(bool from_file, T defval1, T defval2, T val1, T val2)
{
    alps::params p;
    if (from_file) {
        p = get_file_param("with_default", "no_default", val1, val2);
    } else {
        p = get_cmdline_param("with_default", "no_default", val1, val2);
    }
    p.description("This is a test program").
        template define<T>("with_default", defval1, "defined parameter with default").
        template define<T>("no_default", "defined parameter, no default").
        template define<T>("undefined", defval2, "undefined parameter, with default").
        template define<T>("undefined2", "undefined parameter, no default");

    //Access the parameters
    EXPECT_EQ(p["with_default"], val1);
    EXPECT_EQ(p["no_default"], val2);
    EXPECT_EQ(p["undefined"], defval2);

    //Check the "defaulted" status
    EXPECT_FALSE(p.defaulted("with_default"));
    EXPECT_FALSE(p.defaulted("no_default"));
    EXPECT_TRUE(p.defaulted("undefined"));
    
    EXPECT_THROW({const T& x=p["undefined2"]; dummy_use(&x);}, alps::params::uninitialized_value);
}

TEST(param,DefaultsCmdline)
{
    TestDefaults<int>(false, 1,2,3,4);
    TestDefaults<double>(false, 1.25,2.125,4.5,8.0);
    TestDefaults<bool>(false, true,false,false,true);
    TestDefaults<std::string>(false, "def1", "def2", "val1", "val2");
}

TEST(param,DefaultsFromFile)
{
    TestDefaults<int>(true, 1,2,3,4);
    TestDefaults<double>(true, 1.25,2.125,4.5,8.0);
    TestDefaults<bool>(true, true,false,false,true);
    TestDefaults<std::string>(true, "def1", "def2", "val1", "val2");
}

// Test default values for parameter without a command line (or a file)
// (an unlikely, but possible case)
TEST(param,BareDefault)
{
    alps::params p;
    p.description("This is a test program")
        .define<int>("with_default", 123, "int parameter with default");

    EXPECT_EQ(123, int(p["with_default"]));
}
    
