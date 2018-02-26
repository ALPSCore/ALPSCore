/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_eq.cpp
    
    @brief Tests the comparison of parameter objects
*/

#include "./params_test_support.hpp"

#include <alps/params/iniparser_interface.hpp>

using alps::params;

namespace test_data {
    static const char inifile_content[]=
        "my_bool=true\n"
        "my_int=1234\n"
        "my_string=simple\n"
        "my_double=12.75\n"
        ;

    // the same content reordered
    static const char inifile_content_reordered[]=
        "my_int=1234\n"
        "my_string=simple\n"
        "my_bool=true\n"
        "my_double=12.75\n"
        ;

    // the same content truncated
    static const char inifile_content_truncated[]=
        "my_int=1234\n"
        "my_string=simple\n"
        "my_bool=true\n"
        ;
}

class ParamsTest : public ::testing::Test {
  protected:
    ParamsAndFile params_and_file_;
    params& par_;
    const params& cpar_;
  public:
    ParamsTest() : params_and_file_(::test_data::inifile_content),
                    par_(*params_and_file_.get_params_ptr()),
                    cpar_(par_)
    {    }
};

TEST_F(ParamsTest, exactEqual) {
    const params& ref=par_;
    EXPECT_TRUE(cpar_==ref);
}

TEST_F(ParamsTest, reorderEqual) {
    ParamsAndFile data(::test_data::inifile_content_reordered);
    const params& reordered=*data.get_params_ptr();
    EXPECT_TRUE(cpar_==reordered);
}

TEST_F(ParamsTest, fileInequal) {
    ParamsAndFile data(::test_data::inifile_content_truncated);
    const params& another=*data.get_params_ptr();
    EXPECT_FALSE(cpar_==another);
}

TEST_F(ParamsTest, eqContent) {
    ParamsAndFile data(::test_data::inifile_content);
    params& par2=*data.get_params_ptr();

    EXPECT_TRUE(cpar_==par2);

    par_["new_int"]=4321;
    EXPECT_FALSE(cpar_==par2);

    par2["new_int"]=0;
    EXPECT_FALSE(cpar_==par2);

    par2["new_int"]=4321;
    EXPECT_TRUE(cpar_==par2);

    par_["empty"];
    EXPECT_FALSE(cpar_==par2);

    par2["empty"];
    EXPECT_TRUE(cpar_==par2);
}

TEST_F(ParamsTest, eqDefinitions) {
    ParamsAndFile data(::test_data::inifile_content);
    params& par2=*data.get_params_ptr();

    par_.define<int>("my_int", 0, "int value");
    EXPECT_FALSE(cpar_==par2);

    par2.define<int>("my_int", 1111, "int value");
    EXPECT_TRUE(cpar_==par2);
    
    par_.define<int>("no_such_int", 0, "int value");
    EXPECT_FALSE(cpar_==par2);

    par2.define<int>("no_such_int", 1111, "int value");
    EXPECT_FALSE(cpar_==par2);
}
    
TEST_F(ParamsTest, eqDescriptions) {
    ParamsAndFile data(::test_data::inifile_content);
    params& par2=*data.get_params_ptr();

    par_.define<int>("my_int",  "a description");
    par2.define<int>("my_int",  "different description");
    EXPECT_FALSE(cpar_==par2);
}

TEST_F(ParamsTest, eqTypes) {
    ParamsAndFile data(::test_data::inifile_content);
    params& par2=*data.get_params_ptr();

    par_.define<int>("my_int",  "a description");
    par2.define<std::string>("my_int",  "a description");
    EXPECT_FALSE(cpar_==par2);
}

