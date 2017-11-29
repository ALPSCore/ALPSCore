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
