/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_hdf5_with_ini.cpp

    @brief Tests for presence of several H5 and INIs in cmdline
*/

#include "./params_test_support.hpp"
#include <alps/hdf5/archive.hpp>

namespace at = alps::testing;

struct ParamsH5CmdlineTest : public ::testing::Test {
    ini_maker inifile_;
    at::unique_file h5file_;

    static void make_h5_file(const std::string& fname) {
        arg_holder args;
        args
            .add("name1=\"name1 from hdf5\"")
            .add("name2=\"name2 from hdf5\"");

        alps::params p(args.argc(), args.argv());
        p.define<std::string>("name1", "Name1");
        p.define<std::string>("name2", "Name2");

        alps::hdf5::archive ar(fname,alps::hdf5::archive::WRITE);
        ar["/parameters"]=p;
    }

    ParamsH5CmdlineTest() : inifile_("inifile_"),
                            h5file_("h5file_", at::unique_file::REMOVE_AFTER)
    {
        make_h5_file(h5file_.name());

        inifile_
            .add("name2=\"name2 from inifile\"")
            .add("name3=\"name3 from inifile\"");
    }
};


// Two h5 files in the command line
TEST_F(ParamsH5CmdlineTest, checkTwoH5Files) {
    at::unique_file h5file2("another_h5file_", at::unique_file::REMOVE_AFTER);
    make_h5_file(h5file2.name());

    arg_holder args;
    args.add(h5file_.name()).add(inifile_.name()).add(h5file2.name());

    try { // can't use EXPECT_THROW: we want to check the exception object
        alps::params p(args.argc(), args.argv());
        FAIL() << "alps::params ctor is expected to throw; it doesn't";
    } catch (const alps::params::archive_conflict& exc) {
        EXPECT_EQ(h5file_.name(), exc.get_name(0)) << "exception should know 1st archive filename";
        EXPECT_EQ(h5file2.name(), exc.get_name(1)) << "exception should know 2nd archive filename";
        std::string what_msg=exc.what();
        EXPECT_TRUE(what_msg.find(h5file_.name()) != std::string::npos) << "exception message should mention 1st archive filename";
        EXPECT_TRUE(what_msg.find(h5file2.name()) != std::string::npos) << "exception message should mention 2nd archive filename";
        // std::cout << "DEBUG: " << what_msg << std::endl;
    } catch (...) {
        FAIL() << "alps::param ctor threw exception of unexpected type";
    }
}


// Ini file is followed by H5 file
TEST_F(ParamsH5CmdlineTest, checkIniBeforeH5) {
    arg_holder args;
    args.add(inifile_.name()).add(h5file_.name());

    alps::params p(args.argc(), args.argv());

    p.define<std::string>("name1", "Name1");
    p.define<std::string>("name2", "Name2");
    p.define<std::string>("name3", "Name3");

    // INI overrides H5 values
    EXPECT_EQ("name1 from hdf5", p["name1"].as<std::string>());
    EXPECT_EQ("name2 from inifile", p["name2"].as<std::string>());
    EXPECT_EQ("name3 from inifile", p["name3"].as<std::string>());

    // Origin name is archive name
    EXPECT_EQ(h5file_.name(), p.get_archive_name());
    EXPECT_EQ(h5file_.name(), origin_name(p));
}


// H5 file is followed by INI file
TEST_F(ParamsH5CmdlineTest, checkH5BeforeIni) {
    arg_holder args;
    args.add(h5file_.name()).add(inifile_.name());

    alps::params p(args.argc(), args.argv());

    p.define<std::string>("name1", "Name1");
    p.define<std::string>("name2", "Name2");
    p.define<std::string>("name3", "Name3");

    // INI overrides H5 values
    EXPECT_EQ("name1 from hdf5", p["name1"].as<std::string>());
    EXPECT_EQ("name2 from inifile", p["name2"].as<std::string>());
    EXPECT_EQ("name3 from inifile", p["name3"].as<std::string>());

    // Origin name is archive name
    EXPECT_EQ(h5file_.name(), p.get_archive_name());
    EXPECT_EQ(h5file_.name(), origin_name(p));
}
