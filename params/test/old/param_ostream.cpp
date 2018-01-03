/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <sstream>

#include "boost/lexical_cast.hpp"

#include "alps/params.hpp"
#include "gtest/gtest.h"

#include "param_generators.hpp"
#include "alps/utilities/short_print.hpp"
using namespace alps::params_ns::testing;


/* The idea is to generate parameters in different ways (from command
 * line, from default, by explicit assignment), and then check that
 * they are printed correctly. */

/* FIXME: Trigger values are not tested; */
/* FIXME: None-values are not tested; */

/* FIXME: the whole thing looks way too complicated for such a simple task? */

namespace {

    // NOTE: Adjust the functions below if the print format changes
    
    // utility function: stringify a name-value (scalar) from parameter as it would be expected on print
    template <typename T>
    std::string toPrintString(const std::string& name, T val)
    {
        return name + " : " + boost::lexical_cast<std::string>(val);
    }

    // utility function: stringify a name-value (vector) from parameter as it would be expected on print
    template <typename T>
    std::string toPrintString(const std::string& name, const std::vector<T>& vec)
    {
        std::ostringstream s;
        s << alps::short_print(vec);
        return name + " : " + s.str();
    }
}
    
// using type-parametrized test fixture; the type is a alps::params-generating class.
template <typename T>
class ParamTest : public ::testing::Test {
    public:

    void PrintTest() const
    {
        T gen("myparam"); // prepare to generate a parameter with the name "myparam"
        alps::params p=gen.params(); // get the parameter

        std::string expected=toPrintString("myparam", gen.data()); // get the name and the value and print them

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
    OverriddenParameter<bool>,

    CmdlineParameter< std::vector<int> >,
    AssignedParameter< std::vector<int> >,

    CmdlineParameter< std::vector<double> >,
    AssignedParameter< std::vector<double> >,

    CmdlineParameter< std::vector<bool> >,
    AssignedParameter< std::vector<bool> >

    > ParamGenTypes;

TYPED_TEST_CASE(ParamTest, ParamGenTypes);

TYPED_TEST(ParamTest,Print)
{
    this->PrintTest();
}
