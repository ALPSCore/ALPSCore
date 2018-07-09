/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_string_vectors.cpp

    @brief Tests parsing of vectors of strings
*/

#include "./params_test_support.hpp"
#include <alps/params.hpp>

TEST(ParamsTestStringVectors, stringvec) {
    arg_holder args("./progname");
    // Nothing fancy! No quotes! No commas in strings!
    args.add("--strings=str1,str2 with spaces,str3");
    alps::params p(args.argc(), args.argv());
    typedef std::vector<std::string> strvec_t;
    p.define<strvec_t>("strings", "A string vector");
    strvec_t expected={"str1","str2 with spaces","str3"};
    strvec_t actual=p["strings"];
    EXPECT_EQ(expected, actual);
}
