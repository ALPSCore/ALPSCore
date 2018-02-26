/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/utilities/fs/remove_extensions.hpp>
#include <alps/utilities/fs/get_dirname.hpp>
#include <alps/utilities/fs/get_basename.hpp>
#include <alps/utilities/fs/get_extension.hpp>

#include <gtest/gtest.h>


struct TestStrings {
    const char* fullpath;
    const char* dirname;
    const char* filename;
    const char* stem;
    const char* extension;
};

std::ostream& operator<<(std::ostream& s, const TestStrings& t)
{
    s << "{Full path=\"" << t.fullpath
      << "\"; Dir name=\"" << t.dirname
      << "\"; File name=\"" << t.filename
      << "\"; Stem name=\"" << t.stem
      << "\"; Extension=\"" << t.extension
      << "\"}";

    return s;
}

class FilenameOperationsTest : public ::testing::TestWithParam<TestStrings> {
  public:
    std::string fullpath;
    std::string dirname;
    std::string filename;
    std::string stem;
    std::string extension;

    FilenameOperationsTest()
        : fullpath(GetParam().fullpath),
          dirname(GetParam().dirname),
          filename(GetParam().filename),
          stem(GetParam().stem),
          extension(GetParam().extension) {}
};

TEST_P(FilenameOperationsTest,RemoveExtensions) {
    EXPECT_EQ(stem, alps::fs::remove_extensions(fullpath));
}

TEST_P(FilenameOperationsTest,GetBasename) {
    EXPECT_EQ(filename, alps::fs::get_basename(fullpath));
}

TEST_P(FilenameOperationsTest,GetDirname) {
    EXPECT_EQ(dirname, alps::fs::get_dirname(fullpath));
}

TEST_P(FilenameOperationsTest,GetExtension) {
    EXPECT_EQ(extension, alps::fs::get_extension(fullpath));
}

// Each test object is a 4-tuple of C-strings: {path, directory_name, filename, path_with_extensions_stripped, extension}
TestStrings mytests[]={
    {"name",                    "", "name",             "name", ""},
    {"name.ext",                "", "name.ext",         "name", ".ext"},
    {"name.ext.another_ext",    "", "name.ext.another_ext", "name", ".another_ext"},
    {"name.ext.another_ext.third_ext", "", "name.ext.another_ext.third_ext", "name", ".third_ext"},
    {"name.",                   "", "name.", "name", "."},
    {".name",                   "", ".name", "", ".name"},
    {".name.ext",               "", ".name.ext", "", ".ext"},
    {".name.ext.another",       "", ".name.ext.another", "", ".another"},
    {".",                       "", ".", ".", ""}, // special case
    {"..",                      "", "..", "..", ""}, // special case
    {"...",                     "", "...", "..", "."},
    {"....",                    "", "....", "..", "."},
    {"..ext",                   "", "..ext", ".", ".ext"},
    {"...ext",                  "", "...ext", "..", ".ext"},
    {"....ext",                  "", "....ext", "..", ".ext"},

    {"./name", ".", "name", "./name", ""},
    {"./name.ext", ".", "name.ext", "./name", ".ext"},
    {"./name.ext.another_ext", ".", "name.ext.another_ext", "./name", ".another_ext"},
    {"./name.ext.another_ext.third_ext", ".", "name.ext.another_ext.third_ext", "./name", ".third_ext"},
    {"./.name",                   ".", ".name", "./", ".name"},
    {"./.name.ext",               ".", ".name.ext", "./", ".ext"},
    {"./.name.ext.another",       ".", ".name.ext.another", "./", ".another"},
    {"./.",                       ".", ".", "./.", ""},
    {"./..",                      ".", "..", "./..", ""},
    {"./...",                     ".", "...", "./..", "."},
    {"./...ext",                  ".", "...ext", "./..", ".ext"},

    {"relpath/to/name", "relpath/to", "name", "relpath/to/name", ""},
    {"relpath/to/name.ext", "relpath/to", "name.ext", "relpath/to/name", ".ext"},
    {"relpath/to/name.ext.another_ext", "relpath/to", "name.ext.another_ext", "relpath/to/name", ".another_ext"},
    {"relpath/to/name.ext.another_ext.third_ext", "relpath/to", "name.ext.another_ext.third_ext", "relpath/to/name", ".third_ext"},
    {"relpath/to/.name",                   "relpath/to", ".name", "relpath/to/", ".name"},
    {"relpath/to/.name.ext",               "relpath/to", ".name.ext", "relpath/to/", ".ext"},
    {"relpath/to/.name.ext.another",       "relpath/to", ".name.ext.another", "relpath/to/", ".another"},
    {"relpath/to/.",                       "relpath/to", ".", "relpath/to/.", ""},
    {"relpath/to/..",                      "relpath/to", "..", "relpath/to/..", ""},
    {"relpath/to/...",                     "relpath/to", "...", "relpath/to/..", "."},
    {"relpath/to/...ext",                  "relpath/to", "...ext", "relpath/to/..", ".ext"},

    {"./relpath/to/name", "./relpath/to", "name", "./relpath/to/name", ""},
    {"./relpath/to/name.ext", "./relpath/to", "name.ext", "./relpath/to/name", ".ext"},
    {"./relpath/to/name.ext.another_ext", "./relpath/to", "name.ext.another_ext", "./relpath/to/name", ".another_ext"},
    {"./relpath/to/name.ext.another_ext.third_ext", "./relpath/to", "name.ext.another_ext.third_ext", "./relpath/to/name", ".third_ext"},
    {"./relpath/to/.name",                   "./relpath/to", ".name", "./relpath/to/", ".name"},
    {"./relpath/to/.name.ext",               "./relpath/to", ".name.ext", "./relpath/to/", ".ext"},
    {"./relpath/to/.name.ext.another",       "./relpath/to", ".name.ext.another", "./relpath/to/", ".another"},
    {"./relpath/to/.",                       "./relpath/to", ".", "./relpath/to/.", ""},
    {"./relpath/to/..",                      "./relpath/to", "..", "./relpath/to/..", ""},
    {"./relpath/to/...",                     "./relpath/to", "...", "./relpath/to/..", "."},
    {"./relpath/to/...ext",                  "./relpath/to", "...ext", "./relpath/to/..", ".ext"},

    {"/path/to/name", "/path/to", "name", "/path/to/name", ""},
    {"/path/to/name.ext", "/path/to", "name.ext", "/path/to/name", ".ext"},
    {"/path/to/name.ext.another_ext", "/path/to", "name.ext.another_ext", "/path/to/name", ".another_ext"},
    {"/path/to/name.ext.another_ext.third_ext", "/path/to", "name.ext.another_ext.third_ext", "/path/to/name", ".third_ext"},
    {"/path/to/.name",                   "/path/to", ".name", "/path/to/", ".name"},
    {"/path/to/.name.ext",               "/path/to", ".name.ext", "/path/to/", ".ext"},
    {"/path/to/.name.ext.another",       "/path/to", ".name.ext.another", "/path/to/", ".another"},
    {"/path/to/.",                       "/path/to", ".", "/path/to/.", ""},
    {"/path/to/..",                      "/path/to", "..", "/path/to/..", ""},
    {"/path/to/...",                     "/path/to", "...", "/path/to/..", "."},
    {"/path/to/...ext",                  "/path/to", "...ext", "/path/to/..", ".ext"},

    {"/path/some.myext/to/name", "/path/some.myext/to", "name", "/path/some.myext/to/name", ""},
    {"/path/some.myext/to/name.ext", "/path/some.myext/to", "name.ext", "/path/some.myext/to/name", ".ext"},
    {"/path/some.myext/to/name.ext.another_ext", "/path/some.myext/to", "name.ext.another_ext", "/path/some.myext/to/name", ".another_ext"},
    {"/path/some.myext/to/name.ext.another_ext.third_ext", "/path/some.myext/to", "name.ext.another_ext.third_ext", "/path/some.myext/to/name", ".third_ext"},
    {"/path/some.myext/to/.name",                   "/path/some.myext/to", ".name", "/path/some.myext/to/", ".name"},
    {"/path/some.myext/to/.name.ext",               "/path/some.myext/to", ".name.ext", "/path/some.myext/to/", ".ext"},
    {"/path/some.myext/to/.name.ext.another",       "/path/some.myext/to", ".name.ext.another", "/path/some.myext/to/", ".another"},
    {"/path/some.myext/to/.",                       "/path/some.myext/to", ".", "/path/some.myext/to/.", ""},
    {"/path/some.myext/to/..",                      "/path/some.myext/to", "..", "/path/some.myext/to/..", ""},
    {"/path/some.myext/to/...",                     "/path/some.myext/to", "...", "/path/some.myext/to/..", "."},
    {"/path/some.myext/to/...ext",                  "/path/some.myext/to", "...ext", "/path/some.myext/to/..", ".ext"},

    {"/path/to/dir/",                               "/path/to/dir", ".", "/path/to/dir/", ""},
    {"/path/to/dir.ext/",                           "/path/to/dir.ext", ".", "/path/to/dir.ext/", ""},
    {"/short_filename/i",                           "/short_filename", "i", "/short_filename/i", ""},
    {"/short_filename/ii",                          "/short_filename", "ii", "/short_filename/ii", ""},

    {"/in_root.ext",                                "/", "in_root.ext", "/in_root", ".ext"}, // this is how boost::filesystem does it
    {"/",                                           "", "/", "/", ""}, // this is how boost::filesystem does it
    {"",                                            "", "", "", ""}
};

INSTANTIATE_TEST_CASE_P(Test,FilenameOperationsTest,::testing::ValuesIn(mytests));
