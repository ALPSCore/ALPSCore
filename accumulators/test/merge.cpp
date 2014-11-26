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
  acc::accumulator_set measurements1,measurements2;
  measurements1 << A("scalar1")
                << A("scalar2");
  measurements2 << A("scalar1")
                << A("scalar2");

  for (int i = 1; i <= COUNT1; ++i) {
    measurements1["scalar1"] << T(i);
    measurements1["scalar2"] << T(i);
  }
  
  for (int i = 1; i <= COUNT2; ++i) {
    measurements2["scalar1"] << T(i);
    measurements2["scalar2"] << T(i);
  }
  
  acc::accumulator_wrapper& a1=measurements1["scalar1"];
  const acc::accumulator_wrapper& a2=measurements2["scalar1"];

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

  ok=true;
  try {
    measurements1.merge(measurements2);
  } catch (std::exception&) {
    ok=false;
  }
  EXPECT_EQ(ok,is_implemented);
  if (!ok) return;

  EXPECT_EQ(measurements1["scalar1"].count(), COUNT1+COUNT2*2);
  EXPECT_EQ(measurements1["scalar2"].count(), COUNT1+COUNT2);
  const T combined_mean2=0.5*T(COUNT1*(COUNT1+1)+2*COUNT2*(COUNT2+1))/(COUNT1+COUNT2*2);
  EXPECT_EQ(measurements1["scalar1"].mean<T>() , combined_mean2);
  EXPECT_EQ(measurements1["scalar2"].mean<T>() , combined_mean);
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
  TEST(accumulator, merge_feature_vector_ ## A ## _ ## N) {                     \
    test_merge_vector<alps::accumulators::A< std::vector<T> >,T>(is_impl); \
  }                                                              


ALPS_TEST_RUN_MERGE_TEST(MeanAccumulator,double,double,true)
ALPS_TEST_RUN_MERGE_TEST(NoBinningAccumulator,double,double,false)


int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

