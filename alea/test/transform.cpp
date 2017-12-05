#include <alps/alea/mean.hpp>
#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>
#include <alps/alea/convert.hpp>
#include <alps/alea/transform.hpp>
#include <alps/alea/propagation.hpp>

#include "gtest/gtest.h"
#include "gtest_util.hpp"
#include "dataset.hpp"

#include <iterator>
#include <iostream>

TEST(jacobian, linear)
{
    Eigen::MatrixXd tfmat = Eigen::MatrixXd::Random(3, 3);
    alps::alea::linear_transform<double> tf = tfmat;

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
