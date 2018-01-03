/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file predicates_and_actions.cpp
    
    @brief Tests the behaviour of exists(), defined(), defaulted() predicates
    and the corresponding actions (reading, redefinitions)
*/

#include "alps/params.hpp"
#include "gtest/gtest.h"

//Dummy function to imitate use of a variable to supress spurious compiler warnings
static inline void dummy_use(const void*) {}

class ParamsTest :public ::testing::Test {
  public:
    alps::params par;

    ParamsTest()
    {
        const char* argv[]={ "program",
                             "--present_no_default=100",
                             "--present_has_default=200" };
        const int argc=sizeof(argv)/sizeof(*argv);

        par=alps::params(argc,argv);
        par
            .define<long>("present_no_default","Option w/o default")
            .define<long>("present_has_default",999,"Option with default")
            .define<long>("missing_no_default", "Another option w/o default")
            .define<long>("missing_has_default", 300, "Another option with default")
            ;
        par["assigned"]=400L;
    }

    template <typename T>
    T get(const std::string& name) { return par[name]; }

    template <typename T>
    void set(const std::string& name, T val) { par[name]=val; }
};

// Existence predicates:

TEST_F(ParamsTest,Exists)
{
    EXPECT_FALSE(this->par.exists("no_such_name"));
    
    EXPECT_TRUE( this->par.exists("assigned"));
    
    EXPECT_TRUE( this->par.exists("present_no_default"));
    EXPECT_TRUE( this->par.exists("present_has_default"));

    EXPECT_FALSE(this->par.exists("missing_no_default"));
    EXPECT_TRUE( this->par.exists("missing_has_default"));
}

TEST_F(ParamsTest,ExistsTypedExact)
{
    EXPECT_FALSE(this->par.exists<long>("no_such_name"));
    
    EXPECT_TRUE( this->par.exists<long>("assigned"));
    
    EXPECT_TRUE( this->par.exists<long>("present_no_default"));
    EXPECT_TRUE( this->par.exists<long>("present_has_default"));

    EXPECT_FALSE(this->par.exists<long>("missing_no_default"));
    EXPECT_TRUE( this->par.exists<long>("missing_has_default"));
}

TEST_F(ParamsTest,ExistsTypedTruncation)
{
    EXPECT_FALSE(this->par.exists<int>("no_such_name"));
    
    EXPECT_FALSE( this->par.exists<int>("assigned"));
    
    EXPECT_FALSE( this->par.exists<int>("present_no_default"));
    EXPECT_FALSE( this->par.exists<int>("present_has_default"));

    EXPECT_FALSE(this->par.exists<int>("missing_no_default"));
    EXPECT_FALSE( this->par.exists<int>("missing_has_default"));
}

TEST_F(ParamsTest,ExistsTypedExtension)
{
    EXPECT_FALSE(this->par.exists<double>("no_such_name"));
    
    EXPECT_TRUE( this->par.exists<double>("assigned"));
    
    EXPECT_TRUE( this->par.exists<double>("present_no_default"));
    EXPECT_TRUE( this->par.exists<double>("present_has_default"));

    EXPECT_FALSE(this->par.exists<double>("missing_no_default"));
    EXPECT_TRUE( this->par.exists<double>("missing_has_default"));
}

TEST_F(ParamsTest,ExistsTypedIncompatible)
{
    EXPECT_FALSE(this->par.exists<std::string>("no_such_name"));
    
    EXPECT_FALSE(this->par.exists<std::string>("assigned"));
    
    EXPECT_FALSE(this->par.exists<std::string>("present_no_default"));
    EXPECT_FALSE(this->par.exists<std::string>("present_has_default"));

    EXPECT_FALSE(this->par.exists<std::string>("missing_no_default"));
    EXPECT_FALSE(this->par.exists<std::string>("missing_has_default"));
}

// Reading actions (cf. existence predicates)

// (FIXME: also tested elsewhere)
TEST_F(ParamsTest,ReadAttempt)
{
    double x;
    
    EXPECT_THROW(x=this->par["no_such_name"], alps::params::uninitialized_value);
    
    x=this->par["assigned"];
    
    x=this->par["present_no_default"];
    x=this->par["present_has_default"];

    EXPECT_THROW(x=this->par["missing_no_default"], alps::params::uninitialized_value);
    x=this->par["missing_has_default"];

    dummy_use(&x);
}

// (FIXME: also tested elsewhere)
TEST_F(ParamsTest,ReadAttemptTruncation)
{
    EXPECT_THROW(this->get<int>("no_such_name"), alps::params::uninitialized_value);
    
    EXPECT_THROW(this->get<int>("assigned"), alps::params::type_mismatch);

    
    EXPECT_THROW(this->get<int>("present_no_default"), alps::params::type_mismatch);
    EXPECT_THROW(this->get<int>("present_has_default"), alps::params::type_mismatch);

    EXPECT_THROW(this->get<int>("missing_no_default"), alps::params::uninitialized_value);
    EXPECT_THROW(this->get<int>("missing_has_default"), alps::params::type_mismatch);
}

// (FIXME: also tested elsewhere)
TEST_F(ParamsTest,ReadAttemptIncompatible)
{
    EXPECT_THROW(this->get<std::string>("no_such_name"), alps::params::uninitialized_value);
    
    EXPECT_THROW(this->get<std::string>("assigned"), alps::params::type_mismatch);
    
    EXPECT_THROW(this->get<std::string>("present_no_default"), alps::params::type_mismatch);
    EXPECT_THROW(this->get<std::string>("present_has_default"), alps::params::type_mismatch);

    EXPECT_THROW(this->get<std::string>("missing_no_default"), alps::params::uninitialized_value);
    EXPECT_THROW(this->get<std::string>("missing_has_default"), alps::params::type_mismatch);
}


// Defined predicate.

TEST_F(ParamsTest,Defined)
{
    EXPECT_FALSE(this->par.defined("no_such_name"));
    
    EXPECT_TRUE( this->par.defined("assigned"));

    EXPECT_TRUE( this->par.defined("present_no_default"));
    EXPECT_TRUE( this->par.defined("present_has_default"));

    EXPECT_TRUE( this->par.defined("missing_no_default"));
    EXPECT_TRUE( this->par.defined("missing_has_default"));
}

// Redefinition test (cf. defined predicate)

// (FIXME: also tested elsewhere)
TEST_F(ParamsTest,RedefinitionAttempt)
{
    this->par.define<double>("no_such_name","A parameter");
    
    EXPECT_THROW(this->par.define<double>("assigned", "A parameter"), alps::params::extra_definition);

    EXPECT_THROW(this->par.define<double>("present_no_default", "A parameter"), alps::params::double_definition);
    EXPECT_THROW(this->par.define<double>("present_has_default", "A parameter"), alps::params::double_definition);

    EXPECT_THROW(this->par.define<double>("missing_no_default", "A parameter"), alps::params::double_definition);
    EXPECT_THROW(this->par.define<double>("missing_has_default", "A parameter"), alps::params::double_definition);
}    


// Default-value predicate

TEST_F(ParamsTest,Defaulted)
{
    EXPECT_FALSE(this->par.defaulted("no_such_name"));
    // FIXME:Undefined: EXPECT_FALSE( this->par.defaulted("assigned"));
    EXPECT_FALSE(this->par.defaulted("present_no_default"));
    EXPECT_FALSE(this->par.defaulted("present_has_default"));
    EXPECT_FALSE(this->par.defaulted("missing_no_default"));
    EXPECT_TRUE( this->par.defaulted("missing_has_default"));
}

