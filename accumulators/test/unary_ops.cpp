/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"
//This program illustrates a problem with taking an inverse of an observable.

template <typename A>
void test_inversion(void)
{
  typedef typename alps::accumulators::value_type<typename A::accumulator_type>::type T; 
  srand48(0);
  alps::accumulators::accumulator_set measurements;
  measurements<<A("one_half");
  const int samples=1000;

  for(int count=0;count<samples;++count){
      measurements["one_half"]<<drand48();
  }

  alps::accumulators::result_set results(measurements);

  //we have measured about 0.5:
  {
    T xmean=results["one_half"].mean<T>();
    EXPECT_NEAR(xmean, T(0.5), 1.e-1);
  }

  //the inverse of 1/2 should be around two:
  {
    T xmean=results["one_half"].inverse().mean<T>();
    EXPECT_NEAR(xmean, T(2.0), 1.e-1);
  }
  
  //the inverse of 1/2 times 2 should be around four:
  {
    const alps::accumulators::result_wrapper& res=T(2.)/results["one_half"];
    T xmean=res.mean<T>();
    EXPECT_NEAR(xmean, T(4.0), 1.e-1);
  }
}

template <typename A>
void test_negation(void)
{
  typedef typename alps::accumulators::value_type<typename A::accumulator_type>::type T; 
  srand48(0);
  alps::accumulators::accumulator_set measurements;
  measurements<<A("one_half");
  const int samples=1000;

  for(int count=0;count<samples;++count){
      measurements["one_half"]<<drand48();
  }

  alps::accumulators::result_set results(measurements);

  //we have measured about 0.5:
  {
    T xmean=results["one_half"].mean<T>();
    EXPECT_NEAR(xmean, T(0.5), 1.e-1);
  }

  //the negation of 1/2 should be around minus 1/2:
  {
    T xmean=(-results["one_half"]).mean<T>();
    EXPECT_NEAR(xmean, T(-0.5), 1.e-1);
  }
  
  // the difference of exact(1/2) and measured(1/2) should be around zero:
  {
    const alps::accumulators::result_wrapper& res=T(0.5) - results["one_half"];
    T xmean=res.mean<T>();
    EXPECT_NEAR(xmean, T(0.0), 1.e-1);
  }
}

#define MAKE_TEST(atype,dtype,func) TEST(accumulators, func ## _ ## atype ## _ ## dtype) { test_ ## func< alps::accumulators::atype<dtype> >(); } 

MAKE_TEST(FullBinningAccumulator,double,inversion)
MAKE_TEST(LogBinningAccumulator,double,inversion)
MAKE_TEST(NoBinningAccumulator,double,inversion)
MAKE_TEST(MeanAccumulator,double,inversion)

MAKE_TEST(FullBinningAccumulator,double,negation)
MAKE_TEST(LogBinningAccumulator,double,negation)
MAKE_TEST(NoBinningAccumulator,double,negation)
MAKE_TEST(MeanAccumulator,double,negation)

