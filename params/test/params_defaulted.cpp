/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_defaulted.cpp
    
    @brief Tests `params::defaulted()` and related functionality
*/

#include "./params_test_support.hpp"

class ParamsTest0 : public ::testing::Test {
  public:
    arg_holder args;
    alps::params par;

    ParamsTest0() {
        args
            .add("supplied=supplied")
            .add("unreferenced=unreferenced")
            .add("supplied_has_default=supplied_has_defult");
        
        alps::params p(args.argc(), args.argv());
        p["from_dict"]="from_dict";
        p
            .define<std::string>("supplied", "Supplied in cmdline")
            .define<std::string>("supplied_has_default", "supplied_default_value",
                                 "Supplied in cmd line but has default")
            .define<std::string>("not_supplied", "not_supplied_default",
                                 "Not supplied in cmdline and has default");
        EXPECT_TRUE(p.ok()) << "parameter initialization";
        swap(par, p);
    }
};

TEST_F(ParamsTest0, supplied) {
    const alps::params& cpar=par;
    EXPECT_TRUE(cpar.supplied("supplied"));
    EXPECT_TRUE(cpar.supplied("unreferenced"));
    EXPECT_TRUE(cpar.supplied("supplied_has_default"));
    EXPECT_FALSE(cpar.supplied("not_supplied"));
    EXPECT_FALSE(cpar.supplied("from_dict"));
    EXPECT_FALSE(cpar.supplied("nonexistent"));
}

TEST_F(ParamsTest0, defaulted) {
    const alps::params& cpar=par;
    EXPECT_FALSE(cpar.defaulted("supplied"));
    EXPECT_FALSE(cpar.defaulted("unreferenced"));
    EXPECT_FALSE(cpar.defaulted("supplied_has_default"));
    EXPECT_TRUE(cpar.defaulted("not_supplied"));
    EXPECT_TRUE(cpar.defaulted("from_dict"));
    EXPECT_FALSE(cpar.defaulted("nonexistent"));
}
