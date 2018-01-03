/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
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
    CmdlineParamGenerator<int>,
    CmdlineParamGenerator<unsigned int>,
    CmdlineParamGenerator<long>,
    CmdlineParamGenerator<unsigned long>,
    CmdlineParamGenerator<double>
    > CmdlineScalarGenerators;

INSTANTIATE_TYPED_TEST_CASE_P(CmdlineScalarParamTest, AnyParamTest, CmdlineScalarGenerators);

typedef ::testing::Types<
    InifileParamGenerator<bool>,
    InifileParamGenerator<int>,
    InifileParamGenerator<unsigned int>,
    InifileParamGenerator<long>,
    InifileParamGenerator<unsigned long>,
    InifileParamGenerator<double>
    > InifileScalarGenerators;

INSTANTIATE_TYPED_TEST_CASE_P(InifileScalarParamTest, AnyParamTest, InifileScalarGenerators);

typedef ::testing::Types<
    H5ParamGenerator<bool>,
    H5ParamGenerator<int>,
    H5ParamGenerator<unsigned int>,
    H5ParamGenerator<long>,
    H5ParamGenerator<unsigned long>,
    H5ParamGenerator<double>
    > H5ScalarGenerators;

INSTANTIATE_TYPED_TEST_CASE_P(H5ScalarParamTest, AnyParamTest, H5ScalarGenerators);

typedef ::testing::Types<
    CmdlineH5ParamGenerator<bool>,
    CmdlineH5ParamGenerator<int>,
    CmdlineH5ParamGenerator<unsigned int>,
    CmdlineH5ParamGenerator<long>,
    CmdlineH5ParamGenerator<unsigned long>,
    CmdlineH5ParamGenerator<double>
    > CmdlineH5ScalarGenerators;

INSTANTIATE_TYPED_TEST_CASE_P(CmdlineH5ScalarParamTest, AnyParamTest, CmdlineH5ScalarGenerators);

typedef ::testing::Types<
    InifileH5ParamGenerator<bool>,
    InifileH5ParamGenerator<int>,
    InifileH5ParamGenerator<unsigned int>,
    InifileH5ParamGenerator<long>,
    InifileH5ParamGenerator<unsigned long>,
    InifileH5ParamGenerator<double>
    > InifileH5ScalarGenerators;

INSTANTIATE_TYPED_TEST_CASE_P(InifileH5ScalarParamTest, AnyParamTest, InifileH5ScalarGenerators);
