/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_origins.cpp

    @brief Tests parameter origin query methods
*/

#include "./params_test_support.hpp"

#include <alps/params.hpp>

TEST(ParamsTestOrigins, simple) {
    alps::params p;
    EXPECT_TRUE(p.get_argv0().empty());
    EXPECT_EQ(0, p.get_ini_name_count());
    ASSERT_NO_THROW(p.get_ini_name(0));
    EXPECT_TRUE(p.get_ini_name(0).empty());
    EXPECT_TRUE(origin_name(p).empty());
}

TEST(ParamsTestOrigins, cmdlineOnly) {
    arg_holder args("path/to/progname.exe");
    args.add("some=string");
    alps::params p(args.argc(), args.argv());
    EXPECT_EQ("path/to/progname.exe", p.get_argv0());
    EXPECT_EQ(0, p.get_ini_name_count());
    ASSERT_NO_THROW(p.get_ini_name(0));
    EXPECT_TRUE(p.get_ini_name(0).empty());
    EXPECT_EQ("progname.exe",origin_name(p));
}

TEST(ParamsTestOrigins, inifileOnly) {
    ini_maker ini("params_origins.ini.");
    ini.add("some=string");
    alps::params p(ini.name());
    EXPECT_TRUE(p.get_argv0().empty());
    EXPECT_EQ(1, p.get_ini_name_count());
    ASSERT_NO_THROW(p.get_ini_name(0));
    EXPECT_EQ(ini.name(), p.get_ini_name(0));
    ASSERT_NO_THROW(p.get_ini_name(1));
    EXPECT_TRUE(p.get_ini_name(1).empty());
    EXPECT_EQ(ini.name(), origin_name(p));
}

TEST(ParamsTestOrigins, inifileInCmdline) {
    ini_maker ini1("params_origins1.ini.");
    ini1.add("some=string");
    ini_maker ini2("params_origins2.ini.");
    ini2.add("another=string2");

    arg_holder args("path/to/progname.exe");
    args.add(ini1.name());
    args.add("some_other=string3");
    args.add(ini2.name());

    alps::params p(args.argc(), args.argv());
    EXPECT_EQ("path/to/progname.exe", p.get_argv0());
    EXPECT_EQ(2, p.get_ini_name_count());

    ASSERT_NO_THROW(p.get_ini_name(0));
    EXPECT_EQ(ini1.name(), p.get_ini_name(0));

    ASSERT_NO_THROW(p.get_ini_name(1));
    EXPECT_EQ(ini2.name(), p.get_ini_name(1));

    ASSERT_NO_THROW(p.get_ini_name(2));
    EXPECT_TRUE(p.get_ini_name(2).empty());

    EXPECT_EQ(ini1.name(), origin_name(p));
}
