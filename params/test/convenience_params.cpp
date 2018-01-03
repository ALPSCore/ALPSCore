/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/params.hpp>
#include <alps/params/convenience_params.hpp>

#include <gtest/gtest.h>
#include "./params_test_support.hpp"

/// Create a parameters object from a given argv[0], define convenience parameters.
static alps::params make_param(const std::string& argv0)
{
    arg_holder args(argv0);
    alps::params p(args.argc(),args.argv());
    alps::define_convenience_parameters(p);
    return p;
}

TEST(ConvenienceParamsTest, timelimit)
{
    EXPECT_EQ(0u, make_param("/some/path")["timelimit"].as<std::size_t>());
}

TEST(ConvenienceParamsTest, SimpleName)
{
    alps::params p=make_param("progname");
    EXPECT_EQ("progname.out.h5", p["outputfile"].as<std::string>());
    EXPECT_EQ("progname.clone.h5", p["checkpoint"].as<std::string>());
}

TEST(ConvenienceParamsTest, PredefinedOutput)
{
    arg_holder args("progname");
    args.add("outputfile=another_name.out.h5");
    alps::params p(args.argc(), args.argv());
    alps::define_convenience_parameters(p);
    EXPECT_EQ("another_name.out.h5", p["outputfile"].as<std::string>());
    EXPECT_EQ("another_name.clone.h5", p["checkpoint"].as<std::string>());
}

TEST(ConvenienceParamsTest, PredefinedCheckpoint)
{
    arg_holder args("progname");
    args.add("checkpoint=another_name.clone.h5");
    alps::params p(args.argc(), args.argv());
    alps::define_convenience_parameters(p);
    EXPECT_EQ("progname.out.h5", p["outputfile"].as<std::string>());
    EXPECT_EQ("another_name.clone.h5", p["checkpoint"].as<std::string>());
}

TEST(ConvenienceParamsTest, NameExtension)
{
    alps::params p=make_param("progname.exe");
    EXPECT_EQ("progname.out.h5", p["outputfile"].as<std::string>());
    EXPECT_EQ("progname.clone.h5", p["checkpoint"].as<std::string>());
}

TEST(ConvenienceParamsTest, NameDoubleExtension)
{
    alps::params p=make_param("progname.somext.exe");
    EXPECT_EQ("progname.out.h5", p["outputfile"].as<std::string>());
    EXPECT_EQ("progname.clone.h5", p["checkpoint"].as<std::string>());
}

TEST(ConvenienceParamsTest, CurDirName)
{
    alps::params p=make_param("./progname");
    EXPECT_EQ("progname.out.h5", p["outputfile"].as<std::string>());
    EXPECT_EQ("progname.clone.h5", p["checkpoint"].as<std::string>());
}

TEST(ConvenienceParamsTest, CurDirNameExtension)
{
    alps::params p=make_param("./progname.exe");
    EXPECT_EQ("progname.out.h5", p["outputfile"].as<std::string>());
    EXPECT_EQ("progname.clone.h5", p["checkpoint"].as<std::string>());
}

TEST(ConvenienceParamsTest, CurDirNameDoubleExtension)
{
    alps::params p=make_param("./progname.somext.exe");
    EXPECT_EQ("progname.out.h5", p["outputfile"].as<std::string>());
    EXPECT_EQ("progname.clone.h5", p["checkpoint"].as<std::string>());
}

TEST(ConvenienceParamsTest, PathToName)
{
    alps::params p=make_param("/path/to/progname");
    EXPECT_EQ("progname.out.h5", p["outputfile"].as<std::string>());
    EXPECT_EQ("progname.clone.h5", p["checkpoint"].as<std::string>());
}

TEST(ConvenienceParamsTest, PathToNameExtension)
{
    alps::params p=make_param("/path/to/progname.exe");
    EXPECT_EQ("progname.out.h5", p["outputfile"].as<std::string>());
    EXPECT_EQ("progname.clone.h5", p["checkpoint"].as<std::string>());
}

TEST(ConvenienceParamsTest, PathToNameDoubleExtension)
{
    alps::params p=make_param("/path/to/progname.somext.exe");
    EXPECT_EQ("progname.out.h5", p["outputfile"].as<std::string>());
    EXPECT_EQ("progname.clone.h5", p["checkpoint"].as<std::string>());
}

