/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/params.hpp>
#include <alps/params/convenience_params.hpp>

#include <boost/scoped_ptr.hpp>
#include <gtest/gtest.h>

/// Create a parameters object from a given argv[0], define convenience parameters.
static alps::params make_param(const std::string& argv0)
{
    const char* argv[]={argv0.c_str()};
    const int argc=sizeof(argv)/sizeof(*argv);
    alps::params p(argc,argv);
    alps::define_convenience_parameters(p);
    return p;
}

TEST(ConvenienceParamsTest, timelimit)
{
    EXPECT_EQ(0, make_param("/some/path")["timelimit"].as<std::size_t>());
}

TEST(ConvenienceParamsTest, SimpleName)
{
    alps::params p=make_param("progname");
    EXPECT_EQ("progname.out.h5", p["outputfile"].as<std::string>());
    EXPECT_EQ("progname.clone.h5", p["checkpoint"].as<std::string>());
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
