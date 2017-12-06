#include <alps/alea/mean.hpp>
#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>
#include <alps/alea/convert.hpp>
#include <alps/alea/transform.hpp>
#include <alps/alea/transformer.hpp>
#include <alps/alea/propagation.hpp>

#include <alps/testing/near.hpp>
#include "gtest/gtest.h"
#include "dataset.hpp"

#include <iterator>
#include <iostream>

TEST(jacobian, linear)
{
    Eigen::MatrixXd tfmat = Eigen::MatrixXd::Random(3, 3);
    alps::alea::linear_transformer<double> tf = tfmat;

    // The Jacobian is now constant
    Eigen::VectorXd x = Eigen::VectorXd::Zero(3);
    Eigen::MatrixXd jac = alps::alea::jacobian<double>(tf, x, 1.0);

    ALPS_EXPECT_NEAR(tfmat, jac, 1e-6);

    // Try other delta
    jac = alps::alea::jacobian<double>(tf, x, 0.1);
    ALPS_EXPECT_NEAR(tfmat, jac, 1e-6);

    // Try other point
    x << 1, 5, 3;
    jac = alps::alea::jacobian<double>(tf, x, 1.0);
    ALPS_EXPECT_NEAR(tfmat, jac, 1e-6);
}

TEST(types, joinings)
{
    using alps::alea::internal::joined;

    EXPECT_TRUE((std::is_same<
                    alps::alea::mean_result<double>,
                    joined<alps::alea::mean_result<double>,
                           alps::alea::mean_result<double> >::result_type
                >::value));
    EXPECT_TRUE((std::is_same<
                    alps::alea::mean_result<double>,
                    joined<alps::alea::var_result<double>,
                           alps::alea::mean_result<double> >::result_type
                >::value));
    EXPECT_TRUE((std::is_same<
                    alps::alea::mean_result<double>,
                    joined<alps::alea::mean_result<double>,
                           alps::alea::batch_result<double> >::result_type
                >::value));
    EXPECT_TRUE((std::is_same<
                    alps::alea::var_result<double>,
                    joined<alps::alea::cov_result<double>,
                           alps::alea::autocorr_result<double> >::result_type
                >::value));
}

TEST(twogauss, join)
{
    alps::alea::mean_acc<double> acc1;
    for (size_t i = 0; i != twogauss_count; ++i)
        acc1 << twogauss_data[i][0];

    alps::alea::cov_acc<double> acc2;
    for (size_t i = 0; i != twogauss_count; ++i)
        acc2 << twogauss_data[i][1];

    std::vector<double> joined_mean =
                    alps::alea::join(acc1.result(), acc2.result()).mean();
    EXPECT_EQ(2U, joined_mean.size());
    EXPECT_NEAR(twogauss_mean[0], joined_mean[0], 1e-6);
    EXPECT_NEAR(twogauss_mean[1], joined_mean[1], 1e-6);
}

template <typename Acc>
class twogauss_join_case
    : public ::testing::Test
{
public:
    typedef typename alps::alea::traits<Acc>::value_type value_type;
    typedef typename alps::alea::traits<Acc>::result_type result_type;

    twogauss_join_case() { }

    void test_result()
    {
        Acc acc1;
        for (size_t i = 0; i != twogauss_count; ++i)
            acc1 << twogauss_data[i][0];

        Acc acc2;
        for (size_t i = 0; i != twogauss_count; ++i)
            acc2 << twogauss_data[i][1];

        std::vector<double> joined_mean =
                        alps::alea::join(acc1.result(), acc2.result()).mean();
        EXPECT_EQ(2U, joined_mean.size());
        EXPECT_NEAR(twogauss_mean[0], joined_mean[0], 1e-6);
        EXPECT_NEAR(twogauss_mean[1], joined_mean[1], 1e-6);
    }
};

typedef ::testing::Types<
      alps::alea::mean_acc<double>
    , alps::alea::var_acc<double>
    , alps::alea::cov_acc<double>
    , alps::alea::autocorr_acc<double>
    , alps::alea::batch_acc<double>
    > joinable;

TYPED_TEST_CASE(twogauss_join_case, joinable);
TYPED_TEST(twogauss_join_case, test_result) { this->test_result(); }

