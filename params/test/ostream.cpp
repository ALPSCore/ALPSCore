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

#include "param_generators.hpp"
using namespace alps::params_ns::testing;


/* The idea is to generate parameters in different ways (from command
 * line, from default, by explicit assignment), and then check that
 * they are printed correctly. */

/* FIXME: Vectors and trigger values are not tested; */

/* FIXME: the whole thing looks way too complicated for such a simple task? */


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
