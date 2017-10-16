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
        "[section1]\n"
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

TEST_F(ParamsTest0, DISABLED_defineNoDefault) {
    EXPECT_FALSE(cpar_.exists("simple_string"));
    
    par_.define<std::string>("simple_string", "Simple string parameter");
    ASSERT_TRUE(cpar_.exists<std::string>("simple_string"));

    std::string actual=cpar_["simple_string"];
    EXPECT_EQ("simple!", actual);
}
