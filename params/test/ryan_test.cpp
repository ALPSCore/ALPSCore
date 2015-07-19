/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** Test the bug that Ryan reported */

#include "param_generators.hpp"
#include "alps/params.hpp"
#include "gtest/gtest.h"

// GoogleTest fixture: parametrized by parameter generator type
template <typename G>
class ParamTest : public ::testing::Test {
    public:
    G gen;
    typedef typename G::value_type value_type;
    
    ParamTest(): gen("param") {}
    alps::params par() { return gen.params(); }
};

TYPED_TEST_CASE_P(ParamTest);

TYPED_TEST_P(ParamTest,AssignInt)
{
    alps::params p=this->par();
    EXPECT_THROW(p["param"]=int(1),alps::params::type_mismatch);
}

REGISTER_TYPED_TEST_CASE_P(ParamTest,AssignInt);

using namespace alps::params_ns::testing;

typedef ::testing::Types<
    CmdlineParameter<double>,
    CmdlineParameterWithDefault<double>,
    ParameterWithDefault<double>,
    ParameterNoDefault<double>,
    MissingParameterWithDefault<double>,
    MissingParameterNoDefault<double>,
    AssignedParameter<double>,
    OverriddenParameter<double> > double_param_types;

INSTANTIATE_TYPED_TEST_CASE_P(DoubleType, ParamTest, double_param_types);

typedef ::testing::Types<
    CmdlineParameter<std::string>,
    CmdlineParameterWithDefault<std::string>,
    ParameterWithDefault<std::string>,
    ParameterNoDefault<std::string>,
    MissingParameterWithDefault<std::string>,
    MissingParameterNoDefault<std::string>,
    AssignedParameter<std::string>,
    OverriddenParameter<std::string> > string_param_types;

INSTANTIATE_TYPED_TEST_CASE_P(StringType, ParamTest, string_param_types);
