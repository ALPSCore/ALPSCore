/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file vec_scalar_binop_simple.cpp
    Simple test to demonstrate inconsistent behavior of vector-scalar_const binary operations.
*/


#include "gtest/gtest.h"
#include "alps/accumulators.hpp"

// Sum scalar constant of type T2 and accumulator containing vector of T1
template <typename T1, typename T2>
void do_sum()
{
  using namespace alps::accumulators;
  accumulator_set m;
  typedef std::vector<T1> data_type; // vector of T1
  m << NoBinningAccumulator<data_type>("acc"); 
  m["acc"] << data_type(3,0.0);
  result_set res(m);
  try {
    res["acc"]+T2(0.); // adding T2
  } catch (std::runtime_error& exc) {
    FAIL() << std::string("Exception:\n")+exc.what();
  }
}
  
// Add float to vector of floats.
TEST(AccumulatorTest,AddFloatToFloatVec)
{
  do_sum<float,float>();
}

// Add double to vector of floats
TEST(AccumulatorTest,AddDoubleToFloatVec)
{
  do_sum<float,double>();
}

// Add float to vector of doubles
TEST(AccumulatorTest,AddFloatToDoubleVec)
{
  do_sum<double,float>();
}
