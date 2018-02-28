#include <alps/alea.hpp>
#include <alps/alea/util/model.hpp>

#include <alps/testing/near.hpp>
#include "gtest/gtest.h"

#include <iostream>

static const double phi0[] = {2., 3.};
static const double phi1[] = {.5, .25, .25, .3};
static const double veps[] = {1.0, 0.25};

static const double mean[] = {7.47826087, 6.95652174};

alps::alea::util::var1_model<double> get_test_model()
{
    // Model setup
    Eigen::VectorXd phi0(2), veps(2);
    Eigen::MatrixXd phi1(2,2);
    phi0 << 2, 3;
    phi1 << .5, .25, .25, .3;
    veps << 1.0, 0.25;
    std::cerr << "PHI0=\n" << phi0
              << "\nPHI1=\n" << phi1
              << "\nVEPS=\n" << veps << "\n";
    return alps::alea::util::var1_model<double>(phi0, phi1, veps);
}

TEST(var1_test_model, mean)
{
    alps::alea::util::var1_model<double> model = get_test_model();

    // Check mean
    Eigen::VectorXd mean(2);
    mean << 7.47826087, 6.95652174;
    ALPS_EXPECT_NEAR(mean, model.mean(), 1e-6);
}

TEST(var1_test_model, cov)
{
    alps::alea::util::var1_model<double> model = get_test_model();

    // Check covariance
    Eigen::MatrixXd cov(2,2);
    cov << 1.45881536, 0.27152712,
           0.27152712, 0.41967586;
    ALPS_EXPECT_NEAR(cov, model.cov(), 1e-6);
}

TEST(var1_test_model, tau)
{
    alps::alea::util::var1_model<double> model = get_test_model();

    // Check autocorrelation
    Eigen::MatrixXd ctau(2,2);
    ctau << 1.43478261, 0.86956522,
            0.86956522, 0.73913043;
    ALPS_EXPECT_NEAR(ctau, model.ctau(), 1e-6);
}
