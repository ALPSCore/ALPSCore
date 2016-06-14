/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file common_param_tests.cpp
    Generate a parameter object with in a various ways, test conformance to specifications.
    Runs the battery of tests on cmdline-generated parameter object.
*/

#include "common_param_tests.hpp"

typedef ::testing::Types<
    CmdlineParamGenerator<bool>,
    CmdlineParamGenerator<char>,
    CmdlineParamGenerator<int>,
    CmdlineParamGenerator<unsigned int>,
    CmdlineParamGenerator<long>,
    CmdlineParamGenerator<unsigned long>,
    CmdlineParamGenerator<double>
    > CmdlineScalarGenerators;

INSTANTIATE_TYPED_TEST_CASE_P(CmdlineScalarParamTest, AnyParamTest, CmdlineScalarGenerators);
