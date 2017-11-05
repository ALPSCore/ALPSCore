/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_hdf5.cpp
    
    @brief Tests saving/loading of parameters
*/

#include "./params_test_support.hpp"

#include <iostream>

using alps::params;

namespace test_data {
    static const char inifile_content[]=
        "my_bool=true\n"
        "my_int=1234\n"
        "my_string=ABC\n"
        "my_double=12.75\n"
        ;

}

class ParamsTest : public ::testing::Test {
  protected:
    ParamsAndFile params_and_file_;
    params& par_;
    alps::testing::unique_file file_;
  public:
    ParamsTest() : params_and_file_(::test_data::inifile_content),
                   par_(*params_and_file_.get_params_ptr()),
                   file_("params_hdf5_test.h5.", alps::testing::unique_file::KEEP_AFTER)
                   
    {   }
};

TEST_F(ParamsTest, saveLoad) {
    arg_holder args;
    args.add("some=something");
    params p_other(args.argc(), args.argv());
    p_other["another_int"]=9999;

    par_.define<int>("my_int", "Integer param");
    par_.define<double>("my_double", 0.00, "Double param");

    {
        alps::hdf5::archive ar(file_.name(), "w");
        ar["params"] << par_;
    }

    {
        alps::hdf5::archive ar(file_.name(), "r");
        ar["params"] >> p_other;
    }

    EXPECT_FALSE(p_other.exists("another_int"));
    EXPECT_EQ(par_, p_other);

    EXPECT_TRUE(p_other.define<std::string>("my_string", "", "String param").ok());
    EXPECT_EQ("ABC", p_other["my_string"].as<std::string>());
}
