/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params.cpp

    @brief Tests the behaviour of parameters
*/

#include "./params_test_support.hpp"

#include <alps/params/iniparser_interface.hpp>

namespace ap=alps::params_ns;
using ap::dictionary;
using ap::params;
namespace de=ap::exception;
namespace atst=alps::testing;

namespace test_data {
    static const char inifile_content[]=
//        "[HEAD]\n"
        "simple_string=simple!\n"
        "quoted_string=\"quoted\"\n"
        "spaced_string=\"string with spaces\"\n"
        "an_int=1234\n"
        "spaced_int=\"123 456\"\n"
        "quoted_int=\"123456\"\n"
        "# it's a comment\n"
        "duplicate=duplicate1\n"
        "duplicate=duplicate2\n"
        "empty_quoted_string=\"\"\n"
        "empty_string=\n"
        "empty_string_trailing_space= \n"
        "MiXed_CaSe=MiXeD\n"
        "[section1]\n"
        "# it's another comment\n"
        "simple_string=simple1!\n"
        "quoted_string=\"quoted1\"\n"
        "[empty]\n"
        "[section2]\n"
        "simple_string=simple2!\n"
;
}


// FIXME: This class tests implementation details,
//        will likely be removed at some point
class IniparserTest : public ::testing::Test {
  protected:
    ParamsAndFile params_and_file_;
    ap::detail::iniparser parser_;
  public:
    IniparserTest() : params_and_file_(::test_data::inifile_content),
                      parser_(params_and_file_.fname())
    {    }
};

TEST_F(IniparserTest, printAll) {
    ap::detail::iniparser::kv_container_type kvs;
    kvs=parser_();
    for(const ap::detail::iniparser::kv_pair& kv: kvs) {
        std::cout << "Key='" << kv.first << "' value='" << kv.second << "'\n";
    }
}

class ParamsTest0 : public testing::Test {
  protected:
    ParamsAndFile params_and_file_;
    params& par_;
    const params& cpar_;
  public:
    ParamsTest0() : params_and_file_(::test_data::inifile_content),
                    par_(*params_and_file_.get_params_ptr()),
                    cpar_(par_)
    {    }

};

TEST_F(ParamsTest0, ctor) {
    EXPECT_FALSE(cpar_.exists("simple_string"));
}

TEST_F(ParamsTest0, copyCtor) {
    params par2=cpar_;
    EXPECT_EQ(cpar_, par2);
}

TEST_F(ParamsTest0, assignParams) {
    arg_holder args;
    args.add("one=1").add("two=2");
    params par2(args.argc(), args.argv());
    par2.define<int>("one", "One arg");
    par2["some_string"]="some string value";

    par2=cpar_;
    EXPECT_EQ(cpar_, par2);
}

TEST_F(ParamsTest0, swapParams) {
    arg_holder args;
    args.add("one=1").add("two=2");
    params par2(args.argc(), args.argv());
    par2.define<int>("one", "One arg");
    par2["some_string"]="some string value";

    params par2_copy=par2;
    params par1_copy=cpar_;

    using std::swap;
    swap(par_, par2);

    EXPECT_EQ(cpar_, par2_copy);
    EXPECT_EQ(par2, par1_copy);
}



TEST_F(ParamsTest0, quotesAndSpaces) {
    const std::string expected="string with spaces";
    par_.define<std::string>("spaced_string", "Quoted string with spaces");
    const std::string actual=cpar_["spaced_string"];
    EXPECT_EQ(expected, actual);
}

TEST_F(ParamsTest0, numbersWithSpaces) {
    EXPECT_FALSE(par_.define<int>("spaced_int", "Quoted int with spaces").ok());
}

TEST_F(ParamsTest0, numbersWithQuotes) {
    EXPECT_TRUE(par_.define<int>("quoted_int", "Quoted int without spaces").ok());
    const int actual=cpar_["quoted_int"];
    EXPECT_EQ(123456, actual);
}

TEST_F(ParamsTest0, sections) {
    EXPECT_TRUE(par_.define<std::string>("section1.simple_string", "String in sec 1").ok());
    EXPECT_EQ(std::string("simple1!"), cpar_["section1.simple_string"].as<std::string>());
}

TEST_F(ParamsTest0, duplicates) {
    EXPECT_TRUE(par_.define<std::string>("duplicate", "Repeated string").ok());
    EXPECT_EQ(std::string("duplicate2"), cpar_["duplicate"].as<std::string>());
}

TEST_F(ParamsTest0, emptyQuotedString) {
    EXPECT_TRUE(par_.define<std::string>("empty_quoted_string", "Empty quoted string").ok());
    EXPECT_EQ(std::string(), cpar_["empty_quoted_string"].as<std::string>());
}

TEST_F(ParamsTest0, emptyString) {
    EXPECT_TRUE(par_.define<std::string>("empty_string", "Empty string").ok());
    EXPECT_EQ(std::string(), cpar_["empty_string"].as<std::string>());
}

TEST_F(ParamsTest0, emptyStringTrailingSpace) {
    EXPECT_TRUE(par_.define<std::string>("empty_string_trailing_space", "Empty string").ok());
    EXPECT_EQ(std::string(), cpar_["empty_string_trailing_space"].as<std::string>());
}

TEST_F(ParamsTest0, mixedCaseAsMixed) {
    ASSERT_TRUE(par_.define<std::string>("MiXed_CaSe", "default", "Mixed-case").ok());
    EXPECT_EQ("MiXeD", cpar_["MiXed_CaSe"].as<std::string>());
}

TEST_F(ParamsTest0, mixedCaseAsLowercase) {
    ASSERT_TRUE(par_.define<std::string>("mixed_case", "default", "Mixed-case").ok());
    EXPECT_EQ("default", cpar_["mixed_case"].as<std::string>());
}

TEST_F(ParamsTest0, mixedCaseAsUppercase) {
    ASSERT_TRUE(par_.define<std::string>("MIXED_CASE", "default", "Mixed-case").ok());
    EXPECT_EQ("default", cpar_["MIXED_CASE"].as<std::string>());
}


TEST_F(ParamsTest0, flags) {
    arg_holder args;
    args.add("--flag");
    params p(args.argc(), args.argv());
    ASSERT_TRUE(p.define("flag", "A flag option").ok());
    ASSERT_TRUE(p.define("other_flag", "Another flag option").ok());

    EXPECT_TRUE(p["flag"].as<bool>());
    EXPECT_FALSE(p["other_flag"].as<bool>());
}

TEST_F(ParamsTest0, hasMissingRequired) {
    EXPECT_FALSE(par_.define<int>("no_such_int", "whole-number").ok());
    EXPECT_FALSE(par_.define<double>("no_such_double", "floating-point").ok());
    EXPECT_TRUE(par_.has_missing());
    std::ostringstream ostr;
    EXPECT_TRUE(par_.has_missing(ostr));
    EXPECT_TRUE(ostr.str().find("no_such_int")!=std::string::npos);
    // EXPECT_TRUE(ostr.str().find("whole-number")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("no_such_double")!=std::string::npos);
    // EXPECT_TRUE(ostr.str().find("floating-point")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("missing")!=std::string::npos);
    // std::cout << ostr.str(); // DEBUG
}

TEST_F(ParamsTest0, hasMissingParsing) {
    EXPECT_FALSE(par_.define<int>("simple_string", "wrong-number").ok());
    EXPECT_TRUE(par_.has_missing());
    std::ostringstream ostr;
    EXPECT_TRUE(par_.has_missing(ostr));
    EXPECT_TRUE(ostr.str().find("simple_string")!=std::string::npos);
    // EXPECT_TRUE(ostr.str().find("wrong-number")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find(" parse ")!=std::string::npos);
    // std::cout << ostr.str(); // DEBUG
}

TEST_F(ParamsTest0, helpNotRequested) {
    EXPECT_FALSE(par_.help_requested());
    EXPECT_TRUE(par_.exists("help"));

    par_.description("This is a test message");
    EXPECT_TRUE(par_.exists("help"));
    EXPECT_FALSE(par_.help_requested());

    par_.
        define<int>("whole_num", "My-integer").
        define<double>("fp_num", 1.25, "My-fp").
        define<std::string>("solver.name", "Solver name").
        define<double>("solver.precision", 1E-5, "Solver precision").
        define< std::vector<int> >("solver.parameters", "Solver internal parameters");

    std::ostringstream ostr;
    EXPECT_FALSE(par_.help_requested(ostr));
    EXPECT_TRUE(ostr.str().empty());

    EXPECT_EQ(&ostr, & par_.print_help(ostr));
    EXPECT_TRUE(ostr.str().find("This is a test message")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("whole_num")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("My-integer")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("fp_num")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("My-fp")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("1.25")!=std::string::npos);

    std::cout << ostr.str(); // DEBUG
}

TEST_F(ParamsTest0, helpBooleanOff) {
    par_.
        define("some_option", "An option");
    std::ostringstream ostr;
    par_.print_help(ostr);
    EXPECT_TRUE(ostr.str().find("false")!=std::string::npos);
    std::cout << ostr.str(); // DEBUG
}

TEST_F(ParamsTest0, helpBooleanOn) {
    par_.
        define<bool>("some_option", true, "An option normally ON");
    std::ostringstream ostr;
    par_.print_help(ostr);
    EXPECT_TRUE(ostr.str().find("true")!=std::string::npos);
    std::cout << ostr.str(); // DEBUG
}

TEST_F(ParamsTest0, helpRequested) {
    arg_holder args;
    args.add("--help");
    params p(args.argc(), args.argv());

    p.
        description("This is a test message").
        define<int>("whole_num", "My-integer").
        define<double>("fp_num", 1.25, "My-fp").
        define<std::string>("solver.name", "Solver name").
        define<double>("solver.precision", 1E-5, "Solver precision").
        define< std::vector<int> >("solver.parameters", "Solver internal parameters");

    EXPECT_TRUE(p.help_requested());
    std::ostringstream ostr;
    EXPECT_TRUE(p.help_requested(ostr));
    EXPECT_TRUE(ostr.str().find("This is a test message")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("whole_num")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("My-integer")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("fp_num")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("My-fp")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("1.25")!=std::string::npos);
    std::cout << ostr.str(); // DEBUG
}

TEST_F(ParamsTest0, helpRequestedNoDescription) {
    arg_holder args;
    args.add("--help");
    params p(args.argc(), args.argv());

    p.define<int>("whole_num", "My-integer");
    EXPECT_TRUE(p.help_requested());
    std::ostringstream ostr;
    EXPECT_TRUE(p.help_requested(ostr));
    EXPECT_TRUE(ostr.str().find("whole_num")!=std::string::npos);
    std::cout << ostr.str(); // DEBUG
}

TEST_F(ParamsTest0, helpRequestedUserDefined) {
    arg_holder args;
    args.add("--help");
    params p(args.argc(), args.argv());

    EXPECT_TRUE(p.help_requested());

    p
        .define("help", "A user-defined help message")
        .define<int>("whole_num", "My-integer");

    EXPECT_TRUE(p.help_requested());

    std::ostringstream ostr;
    EXPECT_TRUE(p.help_requested(ostr));
    EXPECT_TRUE(ostr.str().find("A user-defined help message")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("whole_num")!=std::string::npos);
    std::cout << ostr.str(); // DEBUG
}

TEST_F(ParamsTest0, helpDeletedRedefined) {
    arg_holder args;
    args.add("--help");
    params p(args.argc(), args.argv());

    // We can reassign a different type, whatever it means
    p["help"]=1234;
    EXPECT_NO_THROW(p["help"].as<int>());
    EXPECT_FALSE(p.help_requested());

    // We can erase help from the dictionary
    p.erase("help");
    EXPECT_FALSE(p.exists("help"));
    EXPECT_FALSE(p.help_requested());

    // but we cannot alter the parameter definition
    EXPECT_THROW(p.define<int>("help", "Help is integer"), de::type_mismatch);

    // and the help message still has the old definition
    // (in this respect all paramnames behave the same)
    p.define<int>("whole_num", "My-integer");

    std::ostringstream ostr;
    p.print_help(ostr);
    EXPECT_FALSE(ostr.str().find("Help is integer")!=std::string::npos);
    EXPECT_TRUE(ostr.str().find("whole_num")!=std::string::npos);
    std::cout << ostr.str(); // DEBUG
}

/* ***** */
/* The following 54 test cases are pre-generated using the script `params_def_gen_test_helper.sh`
   and manually edited.

   The test names are of the following format:

   defined{DEF|NODEF}dict{N|C|W}arg{N|C|W}redef{N|C|W},

   where:
   N generally stand for "nothing", C for "correct", W for "wrong"; specifically:

   defined... : call to the defined<T>():
                { with default | without default }

   dict... : before defined<T>(), the parameter was:
             { not in dictionary | in dictionary, of type T | in dictionary, of other type }

   arg... : in the INI file, the argument is:
            { absent | parseable as T | not parseable as T}

   redef... : the second call to define<X>() with default is:
            { absent | X is T | X is another type }

*/

/*
   Variant 1
   the argument is missing
   not defined in dict
   Parameter defined without default
   not redefined
*/
TEST_F(ParamsTest0, definedNODEFdictNargNredefN) {
    std::string name="no_such_arg";

    /* not in dict */

    EXPECT_EQ("", cpar_.get_descr(name));

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 2
   the argument is missing
   not defined in dict
   Parameter defined without default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedNODEFdictNargNredefC) {
    std::string name="no_such_arg";

    /* not in dict */

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 3
   the argument is missing
   not defined in dict
   Parameter defined without default
   redefined with new type
*/
TEST_F(ParamsTest0, definedNODEFdictNargNredefW) {
    std::string name="no_such_arg";

    /* not in dict */

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    EXPECT_THROW(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
                 de::type_mismatch);

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 4
   the argument is correct type
   not defined in dict
   Parameter defined without default
   not redefined
*/
TEST_F(ParamsTest0, definedNODEFdictNargCredefN) {
    std::string name="an_int";
     const int expected_arg_val=1234;

    /* not in dict */

     EXPECT_TRUE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/*
   Variant 5
   the argument is correct type
   not defined in dict
   Parameter defined without default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedNODEFdictNargCredefC) {
    std::string name="an_int";
    const int expected_arg_val=1234;

    /* not in dict */

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/*
   Variant 6
   the argument is correct type
   not defined in dict
   Parameter defined without default
   redefined with new type
*/
TEST_F(ParamsTest0, definedNODEFdictNargCredefW) {
    std::string name="an_int";
    const int expected_arg_val=1234;

    /* not in dict */

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    EXPECT_THROW(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
                 de::type_mismatch);
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/*
   Variant 7
   the argument is incorrect type
   not defined in dict
   Parameter defined without default
   not redefined
*/
TEST_F(ParamsTest0, definedNODEFdictNargWredefN) {
    std::string name="simple_string";

    /* not in dict */

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 8
   the argument is incorrect type
   not defined in dict
   Parameter defined without default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedNODEFdictNargWredefC) {
    std::string name="simple_string";

    /* not in dict */

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 9
   the argument is incorrect type
   not defined in dict
   Parameter defined without default
   redefined with new type
*/
TEST_F(ParamsTest0, definedNODEFdictNargWredefW) {
    std::string name="simple_string";

    /* not in dict */

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    EXPECT_THROW(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
                 de::type_mismatch);

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 10
   the argument is missing
   pre-defined in dict, same type
   Parameter defined without default
   not redefined
*/
TEST_F(ParamsTest0, definedNODEFdictCargNredefN) {
    std::string name="no_such_arg";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    EXPECT_EQ("", cpar_.get_descr(name));

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(preexisting_int_val, actual);
}

/*
   Variant 11
   the argument is missing
   pre-defined in dict, same type
   Parameter defined without default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedNODEFdictCargNredefC) {
    std::string name="no_such_arg";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(preexisting_int_val, actual);
}

/*
   Variant 12
   the argument is missing
   pre-defined in dict, same type
   Parameter defined without default
   redefined with new type
*/
TEST_F(ParamsTest0, definedNODEFdictCargNredefW) {
    std::string name="no_such_arg";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    EXPECT_THROW(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
                 de::type_mismatch);
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(preexisting_int_val, actual);
}

/*
   Variant 13
   the argument is correct type
   pre-defined in dict, same type
   Parameter defined without default
   not redefined
*/
TEST_F(ParamsTest0, definedNODEFdictCargCredefN) {
    std::string name="an_int";
    const int expected_arg_val=1234;

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/*
   Variant 14
   the argument is correct type
   pre-defined in dict, same type
   Parameter defined without default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedNODEFdictCargCredefC) {
    std::string name="an_int";
     const int expected_arg_val=1234;

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/*
   Variant 15
   the argument is correct type
   pre-defined in dict, same type
   Parameter defined without default
   redefined with new type
*/
TEST_F(ParamsTest0, definedNODEFdictCargCredefW) {
    std::string name="an_int";
    const int expected_arg_val=1234;

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    EXPECT_THROW(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
                 de::type_mismatch);
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/*
   Variant 16
   the argument is incorrect type
   pre-defined in dict, same type
   Parameter defined without default
   not redefined
*/
TEST_F(ParamsTest0, definedNODEFdictCargWredefN) {
    std::string name="simple_string";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 17
   the argument is incorrect type
   pre-defined in dict, same type
   Parameter defined without default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedNODEFdictCargWredefC) {
    std::string name="simple_string";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 18
   the argument is incorrect type
   pre-defined in dict, same type
   Parameter defined without default
   redefined with new type
*/
TEST_F(ParamsTest0, definedNODEFdictCargWredefW) {
    std::string name="simple_string";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default").ok());
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    EXPECT_THROW(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
                 de::type_mismatch);
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 19
   the argument is missing
   pre-defined in dict, different type
   Parameter defined without default
   not redefined
*/
TEST_F(ParamsTest0, definedNODEFdictWargNredefN) {
    std::string name="no_such_arg";

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    EXPECT_THROW(par_.define<int>(name, "Int arg without default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 20
   the argument is missing
   pre-defined in dict, different type
   Parameter defined without default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedNODEFdictWargNredefC) {
    std::string name="no_such_arg";

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    EXPECT_THROW(par_.define<int>(name, "Int arg without default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_THROW(par_.define<int>(name, redef_int_value, "int argument with a default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 21
   the argument is missing
   pre-defined in dict, different type
   Parameter defined without default
   redefined with new type
*/
TEST_F(ParamsTest0, definedNODEFdictWargNredefW) {
    std::string name="no_such_arg";

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    EXPECT_THROW(par_.define<int>(name, "Int arg without default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    EXPECT_TRUE(par_.define<std::string>(name, "NEW default value", "String arg with NEW default").ok());
    EXPECT_EQ("String arg with NEW default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 22
   the argument is correct type
   pre-defined in dict, different type
   Parameter defined without default
   not redefined
*/
TEST_F(ParamsTest0, definedNODEFdictWargCredefN) {
    std::string name="an_int";
    // const int expected_arg_val=1234;

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    EXPECT_THROW(par_.define<int>(name, "Int arg without default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 23
   the argument is correct type
   pre-defined in dict, different type
   Parameter defined without default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedNODEFdictWargCredefC) {
    std::string name="an_int";
    // const int expected_arg_val=1234;

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    EXPECT_THROW(par_.define<int>(name, "Int arg without default"),
                 de::type_mismatch);

    const int redef_int_value=9999;
    EXPECT_THROW(par_.define<int>(name, redef_int_value, "int argument with a default"),
                 de::type_mismatch);

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 24
   the argument is correct type
   pre-defined in dict, different type
   Parameter defined without default
   redefined with new type
*/
TEST_F(ParamsTest0, definedNODEFdictWargCredefW) {
    std::string name="simple_string";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    EXPECT_THROW(par_.define<std::string>(name, "String arg without default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 25
   the argument is incorrect type
   pre-defined in dict, different type
   Parameter defined without default
   not redefined
*/
TEST_F(ParamsTest0, definedNODEFdictWargWredefN) {
    std::string name="simple_string";

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    EXPECT_THROW(par_.define<int>(name, "Int arg without default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 26
   the argument is incorrect type
   pre-defined in dict, different type
   Parameter defined without default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedNODEFdictWargWredefC) {
    std::string name="simple_string";

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    EXPECT_THROW(par_.define<int>(name, "Int arg without default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_THROW(par_.define<int>(name, redef_int_value, "int argument with a default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 27
   the argument is incorrect type
   pre-defined in dict, different type
   Parameter defined without default
   redefined with new type
*/
TEST_F(ParamsTest0, definedNODEFdictWargWredefW) {
    std::string name="simple_string";

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    EXPECT_THROW(par_.define<int>(name, "Int arg without default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    EXPECT_TRUE(par_.define<std::string>(name, "NEW default value", "String arg with NEW default").ok());
    EXPECT_EQ("String arg with NEW default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ("simple!", actual);
}

/*
   Variant 28
   the argument is missing
   not defined in dict
   Parameter defined with default
   not redefined
*/
TEST_F(ParamsTest0, definedDEFdictNargNredefN) {
    std::string name="no_such_arg";

    /* not in dict */

    const int deflt_int_val=1111;
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(deflt_int_val, actual);
}

/*
   Variant 29
   the argument is missing
   not defined in dict
   Parameter defined with default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedDEFdictNargNredefC) {
    std::string name="no_such_arg";

    /* not in dict */

    const int deflt_int_val=1111;
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(deflt_int_val, actual);
}

/*
   Variant 30
   the argument is missing
   not defined in dict
   Parameter defined with default
   redefined with new type
*/
TEST_F(ParamsTest0, definedDEFdictNargNredefW) {
    std::string name="no_such_arg";

    /* not in dict */

    const int deflt_int_val=1111;
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    EXPECT_THROW(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
                 de::type_mismatch);
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(deflt_int_val, actual);
}

/*
   Variant 31
   the argument is correct type
   not defined in dict
   Parameter defined with default
   not redefined
*/
TEST_F(ParamsTest0, definedDEFdictNargCredefN) {
    std::string name="an_int";
     const int expected_arg_val=1234;

    /* not in dict */

    const int deflt_int_val=1111;
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/*
   Variant 32
   the argument is correct type
   not defined in dict
   Parameter defined with default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedDEFdictNargCredefC) {
    std::string name="an_int";
     const int expected_arg_val=1234;

    /* not in dict */

    const int deflt_int_val=1111;
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/*
   Variant 33
   the argument is correct type
   not defined in dict
   Parameter defined with default
   redefined with new type
*/
TEST_F(ParamsTest0, definedDEFdictNargCredefW) {
    std::string name="an_int";
     const int expected_arg_val=1234;

    /* not in dict */

    const int deflt_int_val=1111;
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    EXPECT_THROW(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
                 de::type_mismatch);
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/*
   Variant 34
   the argument is incorrect type
   not defined in dict
   Parameter defined with default
   not redefined
*/
TEST_F(ParamsTest0, definedDEFdictNargWredefN) {
    std::string name="simple_string";

    /* not in dict */

    const int deflt_int_val=1111;
    EXPECT_FALSE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 35
   the argument is incorrect type
   not defined in dict
   Parameter defined with default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedDEFdictNargWredefC) {
    std::string name="simple_string";

    /* not in dict */

    const int deflt_int_val=1111;
    EXPECT_FALSE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 36
   the argument is incorrect type
   not defined in dict
   Parameter defined with default
   redefined with new type
*/
TEST_F(ParamsTest0, definedDEFdictNargWredefW) {
    std::string name="simple_string";

    /* not in dict */

    const int deflt_int_val=1111;
    EXPECT_FALSE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    EXPECT_THROW(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
                 de::type_mismatch);

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 37
   the argument is missing
   pre-defined in dict, same type
   Parameter defined with default
   not redefined
*/
TEST_F(ParamsTest0, definedDEFdictCargNredefN) {
    std::string name="no_such_arg";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    const int deflt_int_val=1111;
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(preexisting_int_val, actual);
}

/*
   Variant 38
   the argument is missing
   pre-defined in dict, same type
   Parameter defined with default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedDEFdictCargNredefC) {
    std::string name="no_such_arg";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    const int deflt_int_val=1111;
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(preexisting_int_val, actual);
}

/*
   Variant 39
   the argument is missing
   pre-defined in dict, same type
   Parameter defined with default
   redefined with new type
*/
TEST_F(ParamsTest0, definedDEFdictCargNredefW) {
    std::string name="no_such_arg";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    const int deflt_int_val=1111;
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    EXPECT_THROW(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
                 de::type_mismatch);
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(preexisting_int_val, actual);
}

/*
   Variant 40
   the argument is correct type
   pre-defined in dict, same type
   Parameter defined with default
   not redefined
*/
TEST_F(ParamsTest0, definedDEFdictCargCredefN) {
    std::string name="an_int";
    const int expected_arg_val=1234;

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    const int deflt_int_val=1111;
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/*
   Variant 41
   the argument is correct type
   pre-defined in dict, same type
   Parameter defined with default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedDEFdictCargCredefC) {
    std::string name="an_int";
    const int expected_arg_val=1234;

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    const int deflt_int_val=1111;
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/*
   Variant 42
   the argument is correct type
   pre-defined in dict, same type
   Parameter defined with default
   redefined with new type
*/
TEST_F(ParamsTest0, definedDEFdictCargCredefW) {
    std::string name="an_int";
     const int expected_arg_val=1234;

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    const int deflt_int_val=1111;
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    EXPECT_THROW(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
                 de::type_mismatch);
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/*
   Variant 43
   the argument is incorrect type
   pre-defined in dict, same type
   Parameter defined with default
   not redefined
*/
TEST_F(ParamsTest0, definedDEFdictCargWredefN) {
    std::string name="simple_string";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    const int deflt_int_val=1111;
    EXPECT_FALSE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 44
   the argument is incorrect type
   pre-defined in dict, same type
   Parameter defined with default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedDEFdictCargWredefC) {
    std::string name="simple_string";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    const int deflt_int_val=1111;
    EXPECT_FALSE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 45
   the argument is incorrect type
   pre-defined in dict, same type
   Parameter defined with default
   redefined with new type
*/
TEST_F(ParamsTest0, definedDEFdictCargWredefW) {
    std::string name="simple_string";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    const int deflt_int_val=1111;
    EXPECT_FALSE(par_.define<int>(name, deflt_int_val, "Int arg with default").ok());
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    EXPECT_THROW(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
                 de::type_mismatch);

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 46
   the argument is missing
   pre-defined in dict, different type
   Parameter defined with default
   not redefined
*/
TEST_F(ParamsTest0, definedDEFdictWargNredefN) {
    std::string name="no_such_arg";

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    const int deflt_int_val=1111;
    EXPECT_THROW(par_.define<int>(name, deflt_int_val, "Int arg with default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 47
   the argument is missing
   pre-defined in dict, different type
   Parameter defined with default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedDEFdictWargNredefC) {
    std::string name="no_such_arg";

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    const int deflt_int_val=1111;
    EXPECT_THROW(par_.define<int>(name, deflt_int_val, "Int arg with default"),
                 de::type_mismatch);

    const int redef_int_value=9999;
    EXPECT_THROW(par_.define<int>(name, redef_int_value, "int argument with a default"),
                 de::type_mismatch);

    EXPECT_EQ("", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 48
   the argument is missing
   pre-defined in dict, different type
   Parameter defined with default
   redefined with new type
*/
TEST_F(ParamsTest0, definedDEFdictWargNredefW) {
    std::string name="no_such_arg";

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    const int deflt_int_val=1111;
    EXPECT_THROW(par_.define<int>(name, deflt_int_val, "Int arg with default"),
                 de::type_mismatch);

    EXPECT_TRUE(par_.define<std::string>(name, "NEW default value", "String arg with NEW default").ok());
    EXPECT_EQ("String arg with NEW default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 49
   the argument is correct type
   pre-defined in dict, different type
   Parameter defined with default
   not redefined
*/
TEST_F(ParamsTest0, definedDEFdictWargCredefN) {
    std::string name="an_int";
    // const int expected_arg_val=1234;

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    const int deflt_int_val=1111;
    EXPECT_THROW(par_.define<int>(name, deflt_int_val, "Int arg with default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 50
   the argument is correct type
   pre-defined in dict, different type
   Parameter defined with default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedDEFdictWargCredefC) {
    std::string name="an_int";
    // const int expected_arg_val=1234;

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    const int deflt_int_val=1111;
    EXPECT_THROW(par_.define<int>(name, deflt_int_val, "Int arg with default"),
                 de::type_mismatch);

    const int redef_int_value=9999;
    EXPECT_THROW(par_.define<int>(name, redef_int_value, "int argument with a default"),
                 de::type_mismatch);

    EXPECT_EQ("", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 51
   the argument is correct type
   pre-defined in dict, different type
   Parameter defined with default
   redefined with new type
*/
TEST_F(ParamsTest0, definedDEFdictWargCredefW) {
    std::string name="simple_string";
    // const std::string expected_arg_val="simple!";

    const int preexisting_int_val=7777;
    par_[name]=preexisting_int_val;

    EXPECT_THROW(par_.define<std::string>(name, "default string val", "String arg with default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default").ok());
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_FALSE(cpar_.exists(name));
}

/*
   Variant 52
   the argument is incorrect type
   pre-defined in dict, different type
   Parameter defined with default
   not redefined
*/
TEST_F(ParamsTest0, definedDEFdictWargWredefN) {
    std::string name="simple_string";

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    const int deflt_int_val=1111;
    EXPECT_THROW(par_.define<int>(name, deflt_int_val, "Int arg with default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 53
   the argument is incorrect type
   pre-defined in dict, different type
   Parameter defined with default
   redefined with the same type
*/
TEST_F(ParamsTest0, definedDEFdictWargWredefC) {
    std::string name="simple_string";

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    const int deflt_int_val=1111;
    EXPECT_THROW(par_.define<int>(name, deflt_int_val, "Int arg with default"),
                 de::type_mismatch);

    const int redef_int_value=9999;
    EXPECT_THROW(par_.define<int>(name, redef_int_value, "int argument with a default"),
                 de::type_mismatch);

    EXPECT_EQ("", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(preexisting_string_val, actual);
}

/*
   Variant 54
   the argument is incorrect type
   pre-defined in dict, different type
   Parameter defined with default
   redefined with new type
*/
TEST_F(ParamsTest0, definedDEFdictWargWredefW) {
    std::string name="simple_string";
    const std::string expected_arg_val="simple!";

    const std::string preexisting_string_val="pre-existing value";
    par_[name]=preexisting_string_val;

    const int deflt_int_val=1111;
    EXPECT_THROW(par_.define<int>(name, deflt_int_val, "Int arg with default"),
                 de::type_mismatch);
    EXPECT_EQ("", cpar_.get_descr(name));

    EXPECT_TRUE(par_.define<std::string>(name, "NEW default value", "String arg with NEW default").ok());
    EXPECT_EQ("String arg with NEW default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/* ***** */
/* *** End of auto-generated tests *** */


// #define X_EXPECT_THROW(a,b) a
