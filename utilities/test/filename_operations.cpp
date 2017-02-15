/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/utilities/remove_extensions.hpp>
#include <alps/utilities/get_dirname.hpp>
#include <alps/utilities/get_basename.hpp>

#include <gtest/gtest.h>


struct TestStrings {
    const char* fullpath;
    const char* dirname;
    const char* filename;
    const char* stem;
};

std::ostream& operator<<(std::ostream& s, const TestStrings& t)
{
    s << "{Full path=\"" << t.fullpath
      << "\"; Dir name=\"" << t.dirname
      << "\"; File name=\"" << t.filename
      << "\"; Stem name=\"" << t.stem
      << "\"}";

    return s;
}

class FilenameOperationsTest : public ::testing::TestWithParam<TestStrings> {
  public:
    std::string fullpath;
    std::string dirname;
    std::string filename;
    std::string stem;

    FilenameOperationsTest()
        : fullpath(GetParam().fullpath),
          dirname(GetParam().dirname),
          filename(GetParam().filename),
          stem(GetParam().stem) {}
  
};

TEST_P(FilenameOperationsTest,RemoveExtensions) {
    EXPECT_EQ(stem, alps::remove_extensions(fullpath));
}

TEST_P(FilenameOperationsTest,GetBasename) {
    EXPECT_EQ(filename, alps::get_basename(fullpath));
}

TEST_P(FilenameOperationsTest,GetDirname) {
    EXPECT_EQ(dirname, alps::get_dirname(fullpath));
}

// Each test object is a 4-tuple of C-strings: {path, directory_name, filename, path_with_extensions_stripped}
TestStrings mytests[]={
    {"name",                    "", "name",             "name"},
    {"name.ext",                "", "name.ext",         "name"},
    {"name.ext.another_ext",    "", "name.ext.another_ext", "name"},
    {"name.ext.another_ext.third_ext", "", "name.ext.another_ext.third_ext", "name"},

    {"./name", ".", "name", "./name"},
    {"./name.ext", ".", "name.ext", "./name"},
    {"./name.ext.another_ext", ".", "name.ext.another_ext", "./name"},
    {"./name.ext.another_ext.third_ext", ".", "name.ext.another_ext.third_ext", "./name"},

    {"/path/to/name", "/path/to", "name", "/path/to/name"},
    {"/path/to/name.ext", "/path/to", "name.ext", "/path/to/name"},
    {"/path/to/name.ext.another_ext", "/path/to", "name.ext.another_ext", "/path/to/name"},
    {"/path/to/name.ext.another_ext.third_ext", "/path/to", "name.ext.another_ext.third_ext", "/path/to/name"},

    {"/path/some.myext/to/name", "/path/some.myext/to", "name", "/path/some.myext/to/name"},
    {"/path/some.myext/to/name.ext", "/path/some.myext/to", "name.ext", "/path/some.myext/to/name"},
    {"/path/some.myext/to/name.ext.another_ext", "/path/some.myext/to", "name.ext.another_ext", "/path/some.myext/to/name"},
    {"/path/some.myext/to/name.ext.another_ext.third_ext", "/path/some.myext/to", "name.ext.another_ext.third_ext", "/path/some.myext/to/name"}
    
};

INSTANTIATE_TEST_CASE_P(Test,FilenameOperationsTest,::testing::ValuesIn(mytests));
