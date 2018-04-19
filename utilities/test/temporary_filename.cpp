/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/utilities/temporary_filename.hpp>
#include <set>

#include <gtest/gtest.h>

TEST(temporary_fname, main)
{
    std::string prefix="./alps_temp_filename_test";
    // NOTE: those files are actually *created*, so be careful with how many of them you generate
    const unsigned int n_uniq_names=(1<<7); // how many uniq names to create

    typedef std::set<std::string> set_type;
    typedef set_type::iterator iter_type;
    typedef std::pair<iter_type,bool> res_type;

    set_type nameset;
    for (unsigned int i=0; i<n_uniq_names; ++i) {
        std::string fname=alps::temporary_filename(prefix);
        ASSERT_EQ(prefix, fname.substr(0,prefix.size()));

        res_type result=nameset.insert(fname);
        ASSERT_TRUE(result.second) << "Clash for i=" << i << " name=" << fname;
    }
}
