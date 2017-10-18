/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params.cpp
    
    @brief Tests the behaviour of parameters
*/

#include <alps/params_new/iniparser_interface.hpp>

#include <alps/params_new.hpp>
#include <alps/testing/unique_file.hpp>

#include <gtest/gtest.h>

#include <fstream>
// #include <boost/scoped_ptr.hpp>
#include <boost/foreach.hpp>

namespace ap=alps::params_new_ns;
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
        "# it's a comment\n"
        "duplicate=duplicate1\n"
        "duplicate=duplicate2\n"
        "[section1]\n"
        "# it's another comment\n"
        "simple_string=simple1!\n"
        "quoted_string=\"quoted1\"\n"
        "[empty]\n"
        "[section2]\n"
        "simple_string=simple2!\n"
;
}

class ParamsAndFile {
    atst::unique_file uniqf_;
    boost::scoped_ptr<params> params_ptr_;

    void write_ini_(const std::string& content) const {
        std::ofstream outf(uniqf_.name().c_str());
        if (!outf) throw std::runtime_error("Can't open temporary file "+uniqf_.name());
        outf << content;
    }

    public:
    // Make param object from a given file content
    ParamsAndFile(const char* ini_content) : uniqf_("params.ini.", atst::unique_file::KEEP_AFTER), params_ptr_(0)
    {
        write_ini_(ini_content);
        params_ptr_.reset(new params(uniqf_.name()));
    }

    const std::string& fname() const { return uniqf_.name(); }
    params* get_params_ptr() const { return params_ptr_.get(); }
};


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
    BOOST_FOREACH(const ap::detail::iniparser::kv_pair& kv, kvs) {
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


TEST_F(ParamsTest0, quotesAndSpaces) {
    const std::string expected="string with spaces";
    par_.define<std::string>("spaced_string", "Quoted string with spaces");
    const std::string actual=cpar_["spaced_string"];
    EXPECT_EQ(expected, actual);
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

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default"));
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

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default"));
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default"));
    EXPECT_EQ("int argument with a default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(redef_int_value, actual);
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

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default"));
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    EXPECT_TRUE(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"));
    EXPECT_EQ("String arg with NEW default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ("NEW default value", actual);
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

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default"));
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

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default"));
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default"));
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

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default"));
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

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default"));
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

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default"));
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default"));
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

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default"));
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    EXPECT_TRUE(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"));
    EXPECT_EQ("String arg with NEW default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ("simple!", actual);
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
    
    EXPECT_TRUE(par_.define<int>(name, "Int arg without default"));
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

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default"));
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default"));
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

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default"));
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

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default"));
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

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default"));
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default"));
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

    EXPECT_TRUE(par_.define<int>(name, "Int arg without default"));
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

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default"));
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    /* not redefined */

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(preexisting_int_val, actual);
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

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default"));
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default"));
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

    EXPECT_FALSE(par_.define<int>(name, "Int arg without default"));
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    EXPECT_THROW(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"),
                 de::type_mismatch);
    EXPECT_EQ("Int arg without default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    int actual=cpar_[name];
    EXPECT_EQ(preexisting_int_val, actual);
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

    EXPECT_TRUE(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"));
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
    EXPECT_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default"));
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

    EXPECT_TRUE(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"));
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
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
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
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default"));
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
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
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
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
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
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default"));
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
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
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
    EXPECT_FALSE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
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
    EXPECT_FALSE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default"));
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
    EXPECT_FALSE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    EXPECT_TRUE(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"));
    EXPECT_EQ("String arg with NEW default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ("simple!", actual);
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
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
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
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default"));
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
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
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
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
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
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_TRUE(par_.define<int>(name, redef_int_value, "int argument with a default"));
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
    EXPECT_TRUE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
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
    EXPECT_FALSE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
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
    EXPECT_FALSE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    const int redef_int_value=9999;
    EXPECT_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default"));
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
    EXPECT_FALSE(par_.define<int>(name, deflt_int_val, "Int arg with default"));
    EXPECT_EQ("Int arg with default", cpar_.get_descr(name));

    EXPECT_TRUE(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"));
    EXPECT_EQ("String arg with NEW default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ("simple!", actual);
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

    EXPECT_TRUE(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"));
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
    EXPECT_FALSE(par_.define<int>(name, redef_int_value, "int argument with a default"));
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

    EXPECT_TRUE(par_.define<std::string>(name, "NEW default value", "String arg with NEW default"));
    EXPECT_EQ("String arg with NEW default", cpar_.get_descr(name));

    ASSERT_TRUE(cpar_.exists(name));
    std::string actual=cpar_[name];
    EXPECT_EQ(expected_arg_val, actual);
}

/* ***** */
/* *** End of auto-generated tests *** */


// #define X_EXPECT_THROW(a,b) a
