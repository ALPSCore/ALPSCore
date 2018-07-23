/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/alea.hpp>

#include "gtest/gtest.h"

#include <iostream>

template <typename Acc>
class empty_mean_case
    : public ::testing::Test
{
public:
    typedef Acc acc_type;
    typedef typename alps::alea::traits<Acc>::store_type  store_type;
    typedef typename alps::alea::traits<Acc>::value_type value_type;
    typedef typename alps::alea::traits<Acc>::result_type result_type;

    void test_zero()
    {
        store_type store(4);

        store.convert_to_mean();
        EXPECT_TRUE(store.data().array().isNaN().all());

        store.convert_to_sum();
        EXPECT_TRUE((store.data().array() == 0).all());
    }
};

typedef ::testing::Types<
      alps::alea::mean_acc<double>
    , alps::alea::mean_acc<std::complex<double> >
    , alps::alea::var_acc<double>
    , alps::alea::var_acc<std::complex<double> >
    , alps::alea::var_acc<std::complex<double>, alps::alea::elliptic_var >
    , alps::alea::cov_acc<double>
    , alps::alea::cov_acc<std::complex<double> >
    , alps::alea::cov_acc<std::complex<double>, alps::alea::elliptic_var >
    > has_mean;

TYPED_TEST_CASE(empty_mean_case, has_mean);

TYPED_TEST(empty_mean_case, test_zero) { this->test_zero(); }


template <typename Acc>
class empty_var_case
    : public ::testing::Test
{
public:
    typedef Acc acc_type;
    typedef typename alps::alea::traits<Acc>::store_type store_type;
    typedef typename alps::alea::traits<Acc>::value_type value_type;
    typedef typename alps::alea::traits<Acc>::result_type result_type;

    void test_zero()
    {
        store_type store(4);

        store.convert_to_mean();
        EXPECT_TRUE(store.data2().array().isNaN().all());

        store.convert_to_sum();
        EXPECT_TRUE((store.data2().array() == 0).all());
    }

    void test_one()
    {
        acc_type acc(4);
        std::array<value_type, 4> vals = {{1., 2., 0., -3.}};
        acc << vals;

        store_type store = acc.store();
        store.convert_to_mean();
        EXPECT_TRUE(store.data2().array().isInf().all());

        store.convert_to_sum();
        EXPECT_TRUE(store.data().isApprox(acc.store().data()));
        EXPECT_TRUE(store.data2().isApprox(acc.store().data2()));
    }
};

typedef ::testing::Types<
      alps::alea::var_acc<double>
    , alps::alea::var_acc<std::complex<double> >
    , alps::alea::cov_acc<double>
    , alps::alea::cov_acc<std::complex<double> >
    > has_var;

TYPED_TEST_CASE(empty_var_case, has_var);

TYPED_TEST(empty_var_case, test_zero) { this->test_zero(); }

TYPED_TEST(empty_var_case, test_one) { this->test_one(); }
