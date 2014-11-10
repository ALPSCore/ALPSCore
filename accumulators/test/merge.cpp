/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators/accumulator.hpp>
#include "gtest/gtest.h"

template<typename A, typename T>
static void test_merge_scalar(bool is_implemented)
{
  namespace acc=alps::accumulators;

  const int COUNT1=1000, COUNT2=500;
  acc::accumulator_set measurements;
  measurements << A("scalar1")
               << A("scalar2");

  for (int i = 1; i <= COUNT1; ++i) {
    measurements["scalar1"] << T(i);
    // EXPECT_EQ(count(measurements["scalar1"]) , i);
    // EXPECT_EQ(measurements["scalar1"].mean<T>(), T(i+1)/2.);
  }
  
  for (int i = 1; i <= COUNT2; ++i) {
    measurements["scalar2"] << T(i);
    // EXPECT_EQ(count(measurements["scalar2"]) , i);
    // EXPECT_EQ(measurements["scalar2"].mean<T>(), T(i+1)/2.);
  }
  
  acc::accumulator_wrapper& a1=measurements["scalar1"];
  const acc::accumulator_wrapper& a2=measurements["scalar2"];

  bool ok=true;
  try {
    a1.merge(a2);
  } catch (std::exception&) {
    ok=false;
  }
  EXPECT_EQ(ok,is_implemented);
  if (!ok) return;

  EXPECT_EQ(count(a1) , COUNT1+COUNT2);
  const T combined_mean=0.5*T(COUNT1*(COUNT1+1)+COUNT2*(COUNT2+1))/(COUNT1+COUNT2);
  EXPECT_EQ(a1.mean<T>() , combined_mean);
}  


template<typename A, typename T>
static void test_merge_vector(bool is_implemented)
{
  typedef std::vector<T> t_vector;
  namespace acc=alps::accumulators;
  
  const int COUNT1=1000, COUNT2=500, LEN=10;
  acc::accumulator_set measurements;
  measurements << A("vector1")
               << A("vector2");

  for (int i = 1; i <= COUNT1; ++i) {
    measurements["vector1"] << t_vector(LEN, T(i));
  }
  
  for (int i = 1; i <= COUNT2; ++i) {
    measurements["vector2"] << t_vector(LEN, T(i));
  }
  
  acc::accumulator_wrapper& a1=measurements["vector1"];
  const acc::accumulator_wrapper& a2=measurements["vector2"];

  bool ok=true;
  try {
    a1.merge(a2);
  } catch (std::exception&) {
    ok=false;
  }
  EXPECT_EQ(ok,is_implemented);
  if (!ok) return;

  EXPECT_EQ(count(a1) , COUNT1+COUNT2);
  const T combined_mean=0.5*T(COUNT1*(COUNT1+1)+COUNT2*(COUNT2+1))/(COUNT1+COUNT2);
  t_vector mean_vec=a1.mean<t_vector>();
  for (int i=0; i<mean_vec.size(); ++i) {
    EXPECT_EQ(mean_vec[i], combined_mean);
  }
}  

#define ALPS_TEST_RUN_MERGE_TEST(A,T,N,is_impl)                       \
  TEST(accumulator, merge_feature_scalar_ ## A ## _ ## N) {           \
    test_merge_scalar<alps::accumulators::A<T>,T>(is_impl);           \
  }                                                                   \
                                                                      \
  TEST(accumulator, merge_feature_vector_ ## A ## N) {                     \
    test_merge_vector<alps::accumulators::A< std::vector<T> >,T>(is_impl); \
  }                                                              


ALPS_TEST_RUN_MERGE_TEST(MeanAccumulator,double,double,true)
ALPS_TEST_RUN_MERGE_TEST(NoBinningAccumulator,double,double,false)

#if 0
TEST(accumulator, merge_feature)
{
  namespace acc=alps::accumulators;
  const int COUNT1=1000, COUNT2=500;
  acc::accumulator_set measurements;
  measurements << acc::MeanAccumulator<double>("scalar1")
               << acc::MeanAccumulator<double>("scalar2");

  for (int i = 1; i <= COUNT1; ++i) {
    measurements["scalar1"] << i;
    EXPECT_EQ(count(measurements["scalar1"]) , i);
    EXPECT_EQ(measurements["scalar1"].mean<double>(), (i+1)/2.0);
  }
  
  for (int i = 1; i <= COUNT2; ++i) {
    measurements["scalar2"] << i;
    EXPECT_EQ(count(measurements["scalar2"]) , i);
    EXPECT_EQ(measurements["scalar2"].mean<double>(), (i+1)/2.0);
  }
  
  acc::accumulator_wrapper& a1=measurements["scalar1"];
  const acc::accumulator_wrapper& a2=measurements["scalar2"];
  
  a1.merge(a2);
  EXPECT_EQ(count(a1) , COUNT1+COUNT2);
  const double combined_mean=0.5*(COUNT1*(COUNT1+1)+COUNT2*(COUNT2+1))/(COUNT1+COUNT2);
  EXPECT_EQ(a1.mean<double>() , combined_mean);
}
#endif

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

