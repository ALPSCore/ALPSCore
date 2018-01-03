/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_cmdline.cpp
    
    @brief Tests parameter input from commandline
*/

// #include <boost/foreach.hpp>

#include "./params_test_support.hpp"

#include <alps/params.hpp>

#include <alps/testing/unique_file.hpp>
#include <gtest/gtest.h>

#include <vector>
#include <fstream>
#include <algorithm>
#include <functional>
#include <locale> // for tolower()

namespace ap=alps::params_ns;
namespace de=ap::exception;
using ap::params;

class ParamsTestCmdline : public ::testing::Test {
  protected:
    arg_holder args_;
  public:
    ParamsTestCmdline() {}
};
    
TEST_F(ParamsTestCmdline, argHolder) {
    ASSERT_EQ(1, args_.argc());
    
    args_.add("arg1").add("arg2");
    ASSERT_EQ(3, args_.argc());

    const char* const* argv=args_.argv();
    EXPECT_EQ(std::string("./program_name"), argv[0]);
    EXPECT_EQ(std::string("arg1"), argv[1]);
    EXPECT_EQ(std::string("arg2"), argv[2]);
}

TEST_F(ParamsTestCmdline, iniMaker) {
    ini_maker maker("inimaker_test.ini");
    maker.add("line1").add("line2");
    std::ifstream infile(maker.name().c_str());
    ASSERT_TRUE(!!infile);
    std::ostringstream ostr;
    ostr << infile.rdbuf();
    EXPECT_EQ(std::string("line1\nline2\n"), ostr.str());
}

// TEST_F(ParamsTestCmdline, originName) {
//     params p0;
//     const params& cp0=p0;
//     EXPECT_EQ("",cp0.get_origin_name());

//     char** argv=0;
//     int argc=0;
//     params p1(argc, argv);
//     EXPECT_EQ("", p1.get_origin_name());

//     params p2(args_.argc(), args_.argv());
//     EXPECT_EQ("./program_name", p2.get_origin_name());
// }


TEST_F(ParamsTestCmdline, filenameArgs) {
    ini_maker ini1("file1.ini"), ini2("file2.ini");

    ini1.add("one=1").add("two=7777");
    ini2.add("two=2").add("three=three");

    args_
        .add(ini1.name())
        .add(ini2.name());

    params p(args_.argc(), args_.argv());

    ASSERT_TRUE(
        p
        .define<int>("one","one")
        .define<int>("two","two")
        .define<std::string>("three","three")
        .ok());

    EXPECT_EQ(1,p["one"]);
    EXPECT_EQ(2,p["two"]);
    EXPECT_EQ("three",p["three"]);
}    

TEST_F(ParamsTestCmdline, boolFlagArgs) {
    args_
        .add("--toggle1")
        .add("-toggle2")
        .add("--sec1:toggle");

    params p(args_.argc(), args_.argv());

    ASSERT_TRUE(p
                .define<bool>("toggle1", false, "Toggle One")
                .define<bool>("toggle2", false, "Toggle Two")
                .define<bool>("sec1:toggle", false, "Toggle in a section")
                .ok());

    EXPECT_TRUE(p["toggle1"].as<bool>());
    EXPECT_TRUE(p["toggle2"].as<bool>());
    EXPECT_TRUE(p["sec1:toggle"].as<bool>());
}

TEST_F(ParamsTestCmdline, keyFlagArgs) {
    args_
        .add("--key1=value1")
        .add("-key2=value2")
        .add("key3=value3")
        .add("key4=");

    params p(args_.argc(), args_.argv());

    ASSERT_TRUE(p
                .define<std::string>("key1", "Key One")
                .define<std::string>("key2", "Key Two")
                .define<std::string>("key3", "Key 3")
                .define<std::string>("key4", "Key 4")
                .ok());

    EXPECT_EQ(p["key1"],"value1");
    EXPECT_EQ(p["key2"],"value2");
    EXPECT_EQ(p["key3"],"value3");
    EXPECT_EQ(p["key4"],"");
}

TEST_F(ParamsTestCmdline, filenamesAndKeys) {
    ini_maker ini1("file1.ini"), ini2("file2.ini");
    ini1.add("one=1").add("two=2").add("zero=0");
    ini2.add("three=3").add("four=4");

    args_
        .add("one=111")    // overrides files even after it
        .add(ini1.name())  // partly overridden by cmdline before and after
        .add("two=222")    // overrides files
        .add(ini2.name())  // partly overridden 
        .add("three=333"); // overrides files

    params p(args_.argc(), args_.argv());
    
    ASSERT_TRUE(p
                .define<int>("zero", "Option 0")
                .define<int>("one", "Option 1")
                .define<int>("two", "Option 2")
                .define<int>("three", "Option 3")
                .define<int>("four", "Option 4")
                .ok());

    EXPECT_EQ(0, p["zero"]) << "option from file1";
    EXPECT_EQ(111, p["one"]) << "option from file1 overridden by cmdine";
    EXPECT_EQ(222, p["two"]) << "option from file1 overridden by cmdline";
    EXPECT_EQ(333, p["three"]) << "option from file2 overridden by cmdline";
    EXPECT_EQ(4, p["four"]) << "option from file2";
}

TEST_F(ParamsTestCmdline, doubleDash) {
    ini_maker ini1("--file1"), ini2("--file2");
    ini1.add("one=1");
    ini2.add("two=2");

    args_
        .add(ini1.name()) // understood as an option
        .add("--") // end of options
        .add(ini2.name()); // understood as a file

    params p(args_.argc(), args_.argv());

    std::string ini1_as_key=ini1.name().substr(2);
    std::string ini2_as_key=ini2.name().substr(2);
    
    ASSERT_TRUE(p
                .define<int>("one", 0, "Option 1")
                .define<int>("two", 0, "Option 2")
                .define<bool>(ini1_as_key, false, "Option that could be file name 1")
                .define<bool>(ini2_as_key, false, "Option that could be file name 2")
                .ok());

    EXPECT_EQ(0, p["one"]) << "one=1 should never be seen";
    EXPECT_EQ(2, p["two"]) << "two=2 should have been read";
    EXPECT_TRUE(!!p[ini1_as_key]) << "--file1... should be taken as option " << ini1_as_key;
    EXPECT_FALSE(!!p[ini2_as_key]) << "--file2... should be taken as file " << ini2_as_key;
}
