/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

// #include <iostream>
// #include <sstream>

#include "alps/params.hpp"
#include "gtest/gtest.h"

// Test for usage of the parameters class within a class-subclass hierarchy

// Global variables: imitation of argc, argv

class BaseClass {
protected:
    alps::params par;

public:
    BaseClass(int argc, const char* argv[]): par(argc, argv)
    {
        par.description("This is the base class").
            define<int>("par1", "Base class parameter1").
            define<int>("par2", "Base class parameter2");
    };

    void do_work(void) const
    {
        int n1=par["par1"];
        EXPECT_EQ(111, n1);

        int n2=par["par2"];
        EXPECT_EQ(222, n2);
    }

    bool check_help(std::ostream& ostr) const
    {
        return par.help_requested(ostr);
    }
};

class DerivedClass: public BaseClass {
public:
    DerivedClass(int argc, const char* argv[]): BaseClass(argc,argv)
    {
        par.description("This is the derived class").
            define<int>("par3","Derived class parameter3").
            define<int>("par4", 444, "Derived class parameter4 with default");
    }

    void do_work(void) const
    {
        BaseClass::do_work();
        
        int n3=par["par3"];
        EXPECT_EQ(333, n3);

        int n4=par["par4"];
        EXPECT_EQ(444, n4);

    }
};

// How the normal functionality works
TEST(param, SubclassWork)
{
    const char* work_argv[]={ "", "--par1=111", "--par2=222", "--par3=333" };
    const int work_argc=sizeof(work_argv)/sizeof(*work_argv);
    std::ostringstream ostr;

    BaseClass base(work_argc, work_argv);
    EXPECT_FALSE(base.check_help(ostr));
    EXPECT_FALSE(ostr.str().find("This is the base class")!=std::string::npos);
    base.do_work();

    ostr.str("");
    DerivedClass derived(work_argc, work_argv);
    EXPECT_FALSE(derived.check_help(ostr));
    EXPECT_FALSE(ostr.str().find("This is the derived class")!=std::string::npos);
    derived.do_work();
}

// How the help request functionality works
TEST(param, SubclassHelp)
{
    const char* help_argv[]={"", "--help"}; 
    const int help_argc=sizeof(help_argv)/sizeof(*help_argv);

    std::ostringstream ostr;
    
    BaseClass base(help_argc, help_argv);
    EXPECT_TRUE(base.check_help(ostr));
    EXPECT_TRUE(ostr.str().find("This is the base class")!=std::string::npos);
    std::cerr << "Base class output:\n" << ostr.str();

    ostr.str("");
    DerivedClass derived(help_argc, help_argv);
    EXPECT_TRUE(derived.check_help(ostr));
    EXPECT_TRUE(ostr.str().find("This is the derived class")!=std::string::npos);
    std::cerr << "\nDerived class output:\n" << ostr.str();
}    

