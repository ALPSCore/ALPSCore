/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <vector>
#include "alps/params.hpp"
#include "gtest/gtest.h"

static void test_assignments()
{
    alps::params parms;
    parms["char"] = static_cast<char>(0x41);
    parms["signed char"] = static_cast<signed char>(0x41);
    parms["unsigned char"] = static_cast<unsigned char>(0x41);
    parms["short"] = static_cast<short>(0x41);
    parms["unsigned short"] = static_cast<unsigned short>(0x41);
    parms["int"] = static_cast<int>(0x41);
    parms["unsigned"] = static_cast<unsigned>(0x41);
    parms["long"] = static_cast<long>(0x41);
    parms["unsigned long"] = static_cast<unsigned long>(0x41);
    parms["long long"] = static_cast<long long>(0x41);
    parms["unsigned long long"] = static_cast<unsigned long long>(0x41);
    parms["float"] = static_cast<float>(0x41);
    parms["double"] = static_cast<double>(0x41);
    parms["long double"] = static_cast<long double>(0x41);
    parms["bool"] = true;
    parms["cstring"] = "asdf";
    parms["std::string"] = std::string("asdf");

    std::vector<double> vd(3);
    vd[0]=1.; vd[1]=2.; vd[2]=4.;
    parms["dblvec"] = vd;
    
    std::cout << std::boolalpha << parms << std::endl;
}

TEST(param, TestParamAssignments){
  test_assignments();
  return;
}
int main(int argc, char **argv) 
{
  // test_assignments();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

