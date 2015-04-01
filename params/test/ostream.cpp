/** Streaming tests */

/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <sstream>

#include "boost/lexical_cast.hpp"

#include "alps/params.hpp"
#include "gtest/gtest.h"



/* The idea is to generate parameters in different ways (from command
 * line, from default, by explicit assignment), and then check that
 * they are printed correctly. */

/* FIXME: Vectors and trigger values are not tested; */

/* FIXME: the whole thing looks way too complicated for such a simple task */

/* FIXME: the code should be moved to a separate header, can be used
 * for testing of MPI and serialization too. */


// using type-parametrized test fixture; the type is a alps::params-generating class.
template <typename T>
class ParamTest : public ::testing::Test {
    public:

    void ScalarPrintTest() const
    {
        typedef typename T::value_type value_type;
        T gen("myparam"); // prepare to generate a parameter with the name "myparam"
        std::string sval=gen.sdata(); // value of the parameter as a string
        alps::params p=gen.params(); // get the parameter

        std::string expected="myparam : "+sval+"\n"; // NOTE: change here if output format changes
        std::ostringstream s;
        s << p;
        EXPECT_TRUE(s.str().find(expected)!=std::string::npos) << "Expected: "+expected+"Got:"+s.str();
    }
};


namespace {
    // Service function: generate a parameter from string as if from a command line
    alps::params gen_param(const std::string& name, const std::string& val)
    {
        std::string arg="--"+name+"="+val;
        const char* argv[]={ "program_name", arg.c_str() };
        const int argc=sizeof(argv)/sizeof(*argv);
        return alps::params(argc,argv);
    }

    // Service functions: generate a test value of type T
    template <typename T>
    inline void gen_data(T& val) { throw std::logic_error("Data generator is not defined"); }

#define ALPS_TEST_DEF_GENERATOR(atype,aval) inline void gen_data(atype& val) { val=aval; }
    ALPS_TEST_DEF_GENERATOR(int,123);
    ALPS_TEST_DEF_GENERATOR(double,124.25);
    ALPS_TEST_DEF_GENERATOR(std::string,"hello, world!");
    ALPS_TEST_DEF_GENERATOR(bool,true);
#undef ALPS_TEST_DEF_GENERATOR

    // Service function: generate "other" test value of type T
    template <typename T>
    inline void gen_other_data(T& val) { gen_data(val); val+=boost::lexical_cast<T>(1); } // works even for strings

    inline void gen_other_data(bool& val) { gen_data(val); val=!val; } 
}


template <typename T>
class BasicParameter
{
    public:
    typedef T value_type;
    std::string name_;
    value_type val_;

    BasicParameter(const std::string& name): name_(name)
    {
        gen_data(val_);
    }
    std::string sdata() const { return boost::lexical_cast<std::string>(val_); } // stored value as string
    T other_data() const { T v; gen_other_data(v); return v; } // data "other than" stored value
};

// Generate a parameter object as if from a command line
template <typename T>
class CmdlineParameter : public BasicParameter<T>
{
    public:
    typedef BasicParameter<T> B;
    CmdlineParameter(const std::string& name): B(name) {}

    alps::params params() const
    {
        alps::params p=gen_param(B::name_, B::sdata());
        return p.define<typename B::value_type>(B::name_,"some parameter");
    }
};

// Generate a parameter object as if from a command line, with the option having a default value
template <typename T>
class CmdlineParameterWithDefault: public BasicParameter<T>
{
    public:
    typedef BasicParameter<T> B;
    CmdlineParameterWithDefault(const std::string& name): B(name) {}

    alps::params params() const
    {
        alps::params p=gen_param(B::name_, B::sdata());
        return p.define<typename B::value_type>(B::name_, B::other_data(), "some parameter");
    }
};

// Generate a parameter object by direct assignment
template <typename T>
class AssignedParameter: public BasicParameter<T>
{
    public:
    typedef BasicParameter<T> B;
    AssignedParameter(const std::string& name): B(name) {}

    alps::params params() const
    {
        alps::params p;
        p[B::name_]=B::val_;
        return p;
    }
};

// Generate a parameter object as if from command line, then assign
template <typename T>
class OverriddenParameter: public BasicParameter<T>
{
    public:
    typedef BasicParameter<T> B;
    OverriddenParameter(const std::string& name): B(name) {}
    alps::params params() const
    {
    alps::params p=gen_param(B::name_, boost::lexical_cast<std::string>(B::other_data()));
        p.define<typename B::value_type>(B::name_, "some parameter");
        p[B::name_]=B::val_;
        return p;
    }
};


typedef ::testing::Types<
    CmdlineParameter<int>,
    CmdlineParameterWithDefault<int>,
    AssignedParameter<int>,
    OverriddenParameter<int>,

    CmdlineParameter<double>,
    CmdlineParameterWithDefault<double>,
    AssignedParameter<double>,
    OverriddenParameter<double>,

    CmdlineParameter<std::string>,
    CmdlineParameterWithDefault<std::string>,
    AssignedParameter<std::string>,
    OverriddenParameter<std::string>,

    CmdlineParameter<bool>,
    CmdlineParameterWithDefault<bool>,
    AssignedParameter<bool>,
    OverriddenParameter<bool>

    > ParamGenTypes;

TYPED_TEST_CASE(ParamTest, ParamGenTypes);

TYPED_TEST(ParamTest,ScalarPrint)
{
    this->ScalarPrintTest();
}
