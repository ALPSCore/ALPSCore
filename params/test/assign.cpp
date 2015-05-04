/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <vector>
#include "alps/params.hpp"
#include "gtest/gtest.h"

#include "param_generators.hpp"

#define ALPS_ASSIGN_PARAM(a_type)  parms[ #a_type ] = static_cast<a_type>(0x41)

#define ALPS_TEST_PARAM(a_type) do { a_type x=parms[ #a_type ]; EXPECT_EQ(x,0x41); } while(0)

TEST(param,assignments)
{
    alps::params parms;
    
    ALPS_ASSIGN_PARAM(char);
    // ALPS_ASSIGN_PARAM(signed char);
    // ALPS_ASSIGN_PARAM(unsigned char);
    // ALPS_ASSIGN_PARAM(short);
    // ALPS_ASSIGN_PARAM(unsigned short);
    ALPS_ASSIGN_PARAM(int);
    ALPS_ASSIGN_PARAM(unsigned);
    ALPS_ASSIGN_PARAM(long);
    ALPS_ASSIGN_PARAM(unsigned long);
    // ALPS_ASSIGN_PARAM(long long);
    // ALPS_ASSIGN_PARAM(unsigned long long);
    // ALPS_ASSIGN_PARAM(float);
    ALPS_ASSIGN_PARAM(double);
    // ALPS_ASSIGN_PARAM(long double);

    parms["bool"] = true;
    parms["cstring"] = "asdf";
    parms["std::string"] = std::string("asdf");

    std::vector<double> vd(3);
    vd[0]=1.; vd[1]=2.; vd[2]=4.;
    parms["dblvec"] = vd;
  
    ALPS_TEST_PARAM(char);
    // ALPS_TEST_PARAM(signed char);
    // ALPS_TEST_PARAM(unsigned char);
    // ALPS_TEST_PARAM(short);
    // ALPS_TEST_PARAM(unsigned short);
    ALPS_TEST_PARAM(int);
    ALPS_TEST_PARAM(unsigned);
    ALPS_TEST_PARAM(long);
    ALPS_TEST_PARAM(unsigned long);
    // ALPS_TEST_PARAM(long long);
    // ALPS_TEST_PARAM(unsigned long long);
    // ALPS_TEST_PARAM(float);
    ALPS_TEST_PARAM(double);
    // ALPS_TEST_PARAM(long double);

    EXPECT_TRUE(bool(parms["bool"]));
    EXPECT_EQ(parms["cstring"],std::string("asdf"));
    EXPECT_EQ(parms["std::string"],std::string("asdf"));

    EXPECT_EQ(parms["dblvec"],vd);
}

// Testing explicitly assigned/implicitly defined parameters and implicit type conversion
TEST(param,ImplicitDefine)
{
    const char* argv[]={"THIS PROGRAM", "--param1=abc", "--param2=4.25"};
    const int argc=sizeof(argv)/sizeof(*argv);

    alps::params p(argc,argv);

    // Explicitly assigning a parameter
    p["param1"]=3.125;
    EXPECT_EQ(p["param1"],3.125);

    {
        // It is not a string, whatever command line is
        EXPECT_THROW(const std::string& s=p["param1"], alps::params::type_mismatch);

        // It can be assigned to an integer
        int n=p["param1"];
        EXPECT_EQ(n, 3);
    }

    // String cannot be assigned to it
    EXPECT_THROW(p["param1"]="abc", alps::params::type_mismatch);

    // Integer cannot be assigned to it either
    EXPECT_THROW(p["param1"]=3, alps::params::type_mismatch);

    // It cannot be defined now, once it was assigned
    EXPECT_THROW((p.define<double>("param1", "Double parameter")), alps::params::extra_definition);

    // Defining another parameter
    p.define<double>("param2", "Another double parameter");

    // Redefinition of the same parameter
    EXPECT_THROW((p.define<double>("param2", "Double parameter again")), alps::params::double_definition);
    EXPECT_THROW((p.define<int>("param2", "Int parameter now")), alps::params::double_definition);

    // Reading the parameter
    EXPECT_EQ(4.25, p["param2"]);
    
    // Assigning to and from the parameter

    // Assign double --- should work
    p["param2"]=5.0;
    EXPECT_EQ(5.0, p["param2"]);

    // Assign int --- should fail
    EXPECT_THROW(p["param2"]=5, alps::params::type_mismatch);
    
    EXPECT_THROW(p["param2"]="abc", alps::params::type_mismatch);
}

// Test object invariants: defining a parameter or parsing a file should not affect the visible state
TEST(param,Invariance)
{
    const char* argv[]={"THIS PROGRAM", "--str=abc", "--int1=111", "--int2=222" };
    const int argc=sizeof(argv)/sizeof(*argv);

    alps::params p(argc,argv);

    p["int3"]=333;

    // Define the string parameter
    p.define<std::string>("str","String parameter");
    // Check the state (triggers parsing!)
    EXPECT_EQ(333, p["int3"]);
    EXPECT_EQ(std::string("abc"), p["str"]);

    // Define the int1 parameter
    p.define<int>("int1","Int parameter1");
    // Check the state (triggers parsing!)
    EXPECT_EQ(std::string("abc"), p["str"]);
    EXPECT_EQ(111, p["int1"]);
    EXPECT_EQ(333, p["int3"]);

    // Change the state
    p["int3"]=-3333;
    p["int1"]=-1111;
    
    // Define the int2 parameter
    p.define<int>("int2","Int parameter2");
    // Check the state (triggers parsing!)
    EXPECT_EQ(222, p["int2"]);
    EXPECT_EQ(std::string("abc"), p["str"]);
    EXPECT_EQ(-1111, p["int1"]);
    EXPECT_EQ(-3333, p["int3"]);

    // Define a non-exisiting parameter
    p.define<int>("int4","Int parameter4 (not provided)");
    // Check the state (triggers parsing!)
    EXPECT_EQ(222, p["int2"]);
    EXPECT_EQ(std::string("abc"), p["str"]);
    EXPECT_EQ(-1111, p["int1"]);
    EXPECT_EQ(-3333, p["int3"]);
}


// FIXME: test type mismatch (on get and on set) systematically for all scalar and vector types


// Testing type mismatch exception and message
TEST(param,SetTypeMismatchMessage)
{
    alps::params_ns::testing::CmdlineParameter<int> gen_int("myparam");
    alps::params p=gen_int.params();

    // FIXME: probably deserve a separate test
    EXPECT_TRUE(  p.exists<int>("myparam") );
    EXPECT_FALSE( p.exists<std::string>("myparam") );
   
    bool thrown=false;
    try {
      p["myparam"]=1L; // attempt to assing long to an integer parameter
    } catch (alps::params::type_mismatch& exc) {
      thrown=true;
      std::string msg=exc.what();
      // std::cerr << "DEBUG: msg='" << msg << "'\n";
      EXPECT_TRUE(msg.find("myparam")!=std::string::npos) << "Option name is not mentioned in exception message: "+msg;
      EXPECT_TRUE(msg.find("int")!=std::string::npos) << "Option type is not mentioned in exception message: "+msg;
      EXPECT_TRUE(msg.find("long")!=std::string::npos) << "RHS type is not mentioned in exception message: "+msg;
    }
    EXPECT_TRUE(thrown) << "Exception was not thrown!";
}

// Testing type mismatch exception and message
TEST(param,GetTypeMismatchMessage)
{
    alps::params_ns::testing::CmdlineParameter<unsigned long> gen_ulong("myparam");
    alps::params p=gen_ulong.params();

    bool thrown=false;
    try {
      std::string s=p["myparam"]; // attempt to assing integer parameter to a string
    } catch (alps::params::type_mismatch& exc) {
      thrown=true;
      std::string msg=exc.what();
      // std::cerr << "DEBUG: msg='" << msg << "'\n";
      EXPECT_TRUE(msg.find("myparam")!=std::string::npos) << "Option name is not mentioned in exception message: "+msg;
      EXPECT_TRUE(msg.find("unsigned long int")!=std::string::npos) << "Option type is not mentioned in exception message: "+msg;
      EXPECT_TRUE(msg.find("std::string")!=std::string::npos) << "LHS type is not mentioned in exception message: "+msg;
    }
    EXPECT_TRUE(thrown) << "Exception was not thrown!";
}
