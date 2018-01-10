/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_has_unused.cpp
    
    @brief Tests checking for unused arguments
*/

#include "./params_test_support.hpp"
#include <sstream>

class ParamsTest0 : public ::testing::Test {
  public:
    ini_maker ini;
    alps::params par;

    ParamsTest0() : ini("params_has_unused.ini.") {
        ini
            .add("supplied=supplied")
            .add("unreferenced1=value_1")
            .add("unreferenced2=value_2")
            .add("supplied_has_default=supplied_has_default")
            .add("[subsection]")
            .add("unreferenced3=value_3")
            .add("unreferenced4=value_4");

        arg_holder args;
        args.add(ini.name());
        
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

TEST_F(ParamsTest0, unusedAll) {
    const alps::params& cpar=par;

    std::ostringstream ostrm;
    EXPECT_TRUE(cpar.has_unused(ostrm));

    std::cout << "DEBUG: output is:\n" << ostrm.str() << "\n===\n";
    
    EXPECT_TRUE(ostrm.str().find("unreferenced1")!=std::string::npos);
    EXPECT_TRUE(ostrm.str().find("unreferenced2")!=std::string::npos);
    EXPECT_TRUE(ostrm.str().find("unreferenced4")!=std::string::npos);
    EXPECT_TRUE(ostrm.str().find("unreferenced3")!=std::string::npos);
    EXPECT_TRUE(ostrm.str().find("subsection")!=std::string::npos);
}

TEST_F(ParamsTest0, unusedAllReferenced) {
    const alps::params& cpar=par;

    par
        .define<std::string>("unreferenced1","some description")
        .define<std::string>("unreferenced2","some description")
        .define<std::string>("subsection.unreferenced3","some description")
        .define<std::string>("subsection.unreferenced4","some description");

    std::ostringstream ostrm;
    EXPECT_FALSE(cpar.has_unused(ostrm));
    EXPECT_TRUE(ostrm.str().empty());
}

TEST_F(ParamsTest0, unusedTop) {
    const alps::params& cpar=par;

    std::ostringstream ostrm;
    EXPECT_TRUE(cpar.has_unused(ostrm,""));

    std::cout << "DEBUG: output is:\n" << ostrm.str() << "\n===\n";
    
    EXPECT_TRUE(ostrm.str().find("unreferenced1")!=std::string::npos);
    EXPECT_TRUE(ostrm.str().find("unreferenced2")!=std::string::npos);
    EXPECT_FALSE(ostrm.str().find("unreferenced3")!=std::string::npos);
    EXPECT_FALSE(ostrm.str().find("unreferenced4")!=std::string::npos);
    EXPECT_FALSE(ostrm.str().find("subsection")!=std::string::npos);
}

TEST_F(ParamsTest0, unusedTopReferenced) {
    const alps::params& cpar=par;

    par
        .define<std::string>("unreferenced1","some description")
        .define<std::string>("unreferenced2","some description");
    
    std::ostringstream ostrm;
    EXPECT_FALSE(cpar.has_unused(ostrm,""));
    EXPECT_TRUE(ostrm.str().empty());
}

TEST_F(ParamsTest0, unusedSection) {
    const alps::params& cpar=par;

    std::ostringstream ostrm;
    EXPECT_TRUE(cpar.has_unused(ostrm,"subsection"));

    std::cout << "DEBUG: output is:\n" << ostrm.str() << "\n===\n";
    
    EXPECT_FALSE(ostrm.str().find("unreferenced1")!=std::string::npos);
    EXPECT_FALSE(ostrm.str().find("unreferenced2")!=std::string::npos);
    EXPECT_TRUE(ostrm.str().find("unreferenced3")!=std::string::npos);
    EXPECT_TRUE(ostrm.str().find("unreferenced4")!=std::string::npos);
    EXPECT_TRUE(ostrm.str().find("subsection")!=std::string::npos);
}

TEST_F(ParamsTest0, unusedSectionReferenced) {
    const alps::params& cpar=par;

    par
        .define<std::string>("subsection.unreferenced3","some description")
        .define<std::string>("subsection.unreferenced4","some description");
    
    std::ostringstream ostrm;
    EXPECT_FALSE(cpar.has_unused(ostrm,"subsection"));
    EXPECT_TRUE(ostrm.str().empty());
}
