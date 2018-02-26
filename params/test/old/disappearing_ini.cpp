/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <fstream>
#include <unistd.h>

#include <alps/params.hpp>
#include <alps/testing/unique_file.hpp>

#include <gtest/gtest.h>

struct ParamTest : public ::testing::Test {
    std::string fname_;
    alps::params params_;

    ParamTest() {
        std::string fn=alps::testing::temporary_filename("./pfile.ini.");
        {
            std::ofstream inifile(fn.c_str());
            if (!inifile) throw std::runtime_error("Cannot open "+fn);
            fname_=fn;

            inifile << "intpar = 1234\n";
        }
        params_=alps::params(fn);
        params_.define<int>("intpar","Integer parameter");
        params_.define<int>("intpar2", 0, "Another integer parameter");
    }

    void remove_ini_file() {
        if (!fname_.empty()) {
            std::remove(fname_.c_str());
        }
        fname_.clear();
    }
    
    ~ParamTest() {
        remove_ini_file();
    }
};


TEST_F(ParamTest,ChdirAfterAccess) {
    params_["intpar2"];
    int rc=chdir("..");
    if (rc!=0) throw std::runtime_error("Cannot change to parent directory");
    int intpar=params_["intpar"];
    EXPECT_EQ(1234, intpar);
}    

TEST_F(ParamTest,ChdirBeforeAccess) {
    int rc=chdir("..");
    if (rc!=0) throw std::runtime_error("Cannot change to parent directory");
    int intpar=params_["intpar"];
    EXPECT_EQ(1234, intpar);
}    

TEST_F(ParamTest,RemoveBeforeAccess) {
    remove_ini_file();
    int intpar=params_["intpar"];
    EXPECT_EQ(1234, intpar);
}    

TEST_F(ParamTest,RemoveAfterAccess) {
    params_["intpar2"];
    remove_ini_file();
    int intpar=params_["intpar"];
    EXPECT_EQ(1234, intpar);
}    

