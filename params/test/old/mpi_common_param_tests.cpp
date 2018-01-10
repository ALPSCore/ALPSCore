/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file common_param_tests.cpp
    Generate a parameter object with in a various ways, test conformance to specifications.
    Runs the battery of tests on cmdline-generated-broadcasted parameter object.
*/

#include "common_param_tests.hpp"
#include "alps/utilities/gtest_par_xml_output.hpp"

typedef ::testing::Types<
    CmdlineMpiParamGenerator<bool>,
    CmdlineMpiParamGenerator<int>,
    CmdlineMpiParamGenerator<unsigned int>,
    CmdlineMpiParamGenerator<long>,
    CmdlineMpiParamGenerator<unsigned long>,
    CmdlineMpiParamGenerator<double>
    > CmdlineMpiScalarGenerators;

INSTANTIATE_TYPED_TEST_CASE_P(CmdlineMpiScalarParamTest, AnyParamTest, CmdlineMpiScalarGenerators);

int main(int argc, char** argv)
{
   alps::mpi::environment env(argc, argv, false);
   alps::gtest_par_xml_output tweak;
   tweak(alps::mpi::communicator().rank(), argc, argv);
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}    
