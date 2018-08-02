/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/alea.hpp>
#include <alps/testing/near.hpp>

#include "gtest/gtest.h"

template <typename Acc>
class result_case
    : public ::testing::Test
{
public:
    typedef Acc acc_type;
    typedef alps::alea::traits<Acc> acc_traits;
    typedef typename acc_traits::store_type  store_type;
    typedef typename acc_traits::value_type value_type;

    typedef typename acc_traits::result_type result_type;
    typedef alps::alea::traits<result_type> result_traits;

    result_case()
    {
        acc_type acc(2);
        for (size_t i = 0; i != 100; ++i)
            acc << std::array<value_type, 2>{{1.0 * i, 2.0 * i}};

        res = acc.result();
        generic_res = acc.result();
    }

    void test_result_caps()
    {
        EXPECT_EQ(res.size(), generic_res.size());
        EXPECT_EQ(res.count(), generic_res.count());
        ALPS_EXPECT_NEAR(res.mean(), generic_res.mean<value_type>(), 1e-16);
    }

    // Thank you, C++, for this very nice way of doing "if/else".  Go die.

    template <typename Res_>
    typename std::enable_if<alps::alea::traits<Res_>::HAVE_VAR>::type test_var()
    {
        typedef typename alps::alea::traits<Res_>::value_type value_type;
        ALPS_EXPECT_NEAR(res.var(), generic_res.var<value_type>(), 1e-16);
    }

    template <typename Res_>
    typename std::enable_if<!alps::alea::traits<Res_>::HAVE_VAR>::type test_var()
    {
        typedef typename alps::alea::traits<Res_>::value_type value_type;
        EXPECT_THROW(generic_res.var<value_type>(), alps::alea::estimate_unavailable);
    }

    template <typename Res_>
    typename std::enable_if<alps::alea::traits<Res_>::HAVE_COV>::type test_cov()
    {
        typedef typename alps::alea::traits<Res_>::value_type value_type;
        ALPS_EXPECT_NEAR(res.cov(), generic_res.cov<value_type>(), 1e-16);
    }

    template <typename Res_>
    typename std::enable_if<!alps::alea::traits<Res_>::HAVE_COV>::type test_cov()
    {
        typedef typename alps::alea::traits<Res_>::value_type value_type;
        EXPECT_THROW(generic_res.cov<value_type>(), alps::alea::estimate_unavailable);
    }


private:
    result_type res;
    alps::alea::result generic_res;
};

typedef ::testing::Types<
      alps::alea::mean_acc<double>
    , alps::alea::mean_acc<std::complex<double> >
    , alps::alea::var_acc<double>
    , alps::alea::var_acc<std::complex<double> >
    , alps::alea::cov_acc<double>
    , alps::alea::cov_acc<std::complex<double> >
    , alps::alea::batch_acc<double>
    , alps::alea::batch_acc<std::complex<double> >
    > acc_types;


TYPED_TEST_CASE(result_case, acc_types);

TYPED_TEST(result_case, test_result_caps) { this->test_result_caps(); }

TYPED_TEST(result_case, test_var) {
    this->template test_var<typename TestFixture::result_type>();
}

TYPED_TEST(result_case, test_cov) {
    this->template test_cov<typename TestFixture::result_type>();
}
