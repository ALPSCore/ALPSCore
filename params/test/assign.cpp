/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <vector>
#include "alps/params.hpp"
#include "gtest/gtest.h"

#define ALPS_ASSIGN_PARAM(a_type)  parms[ #a_type ] = static_cast<a_type>(0x41)

#define ALPS_TEST_PARAM(a_type) do { a_type x=parms[ #a_type ]; EXPECT_EQ(x,0x41); } while(0)

TEST(param,assignments)
{
    alps::params parms;
    
    ALPS_ASSIGN_PARAM(char);
    ALPS_ASSIGN_PARAM(signed char);
    ALPS_ASSIGN_PARAM(unsigned char);
    ALPS_ASSIGN_PARAM(short);
    ALPS_ASSIGN_PARAM(unsigned short);
    ALPS_ASSIGN_PARAM(int);
    ALPS_ASSIGN_PARAM(unsigned);
    ALPS_ASSIGN_PARAM(long);
    ALPS_ASSIGN_PARAM(unsigned long);
    ALPS_ASSIGN_PARAM(long long);
    ALPS_ASSIGN_PARAM(unsigned long long);
    ALPS_ASSIGN_PARAM(float);
    ALPS_ASSIGN_PARAM(double);
    ALPS_ASSIGN_PARAM(long double);

    parms["bool"] = true;
    parms["cstring"] = "asdf";
    parms["std::string"] = std::string("asdf");

    std::vector<double> vd(3);
    vd[0]=1.; vd[1]=2.; vd[2]=4.;
    parms["dblvec"] = vd;
  
    ALPS_TEST_PARAM(char);
    ALPS_TEST_PARAM(signed char);
    ALPS_TEST_PARAM(unsigned char);
    ALPS_TEST_PARAM(short);
    ALPS_TEST_PARAM(unsigned short);
    ALPS_TEST_PARAM(int);
    ALPS_TEST_PARAM(unsigned);
    ALPS_TEST_PARAM(long);
    ALPS_TEST_PARAM(unsigned long);
    ALPS_TEST_PARAM(long long);
    ALPS_TEST_PARAM(unsigned long long);
    ALPS_TEST_PARAM(float);
    ALPS_TEST_PARAM(double);
    ALPS_TEST_PARAM(long double);

    EXPECT_TRUE(bool(parms["bool"]));
    EXPECT_EQ(parms["cstring"],std::string("asdf"));
    EXPECT_EQ(parms["std::string"],std::string("asdf"));

    EXPECT_EQ(parms["dblvec"],vd);

    // FIXME!!! Not yet implemented!!!
    // std::cout << std::boolalpha << parms << std::endl;
}
   

int main(int argc, char **argv) 
{
  // test_assignments();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

