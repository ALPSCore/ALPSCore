#include <alps/alea.hpp>
#include <alps/alea/util/model.hpp>

#include <alps/testing/near.hpp>
#include "gtest/gtest.h"

#include <boost/random/mersenne_twister.hpp>
#include <iostream>

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

template <typename T, typename Acc>
void fill(const alps::alea::util::var1_model<T> &model, Acc &acc, size_t tmax)
{
    // fill accumulator with values from model
    alps::alea::util::var1_run<T> run = model.start();
    boost::random::mt19937 engine;
    while (run.t() < tmax) {
        run.step(engine);
        acc << run.xt();
    }
}

TEST(var1_test, autocorr)
{
    Eigen::VectorXd phi0(2), veps(2);
    Eigen::MatrixXd phi1(2,2);
    phi0 << 2, 3;
    phi1 << .80, 0, 0, .64;
    veps << 1.0, 0.25;
    alps::alea::util::var1_model<double> model(phi0, phi1, veps);

    std::cerr << "EXACT MEAN=" << model.mean().transpose() << "\n";
    std::cerr << "EXACT TAU =" << model.ctau().diagonal().transpose() << "\n";

    alps::alea::autocorr_acc<double> acc(2);

    fill(model, acc, 400000);
    alps::alea::autocorr_result<double> res = acc.finalize();
    std::cerr << "EST.  MEAN=" << res.mean().transpose() << "\n";
    std::cerr << "EST.  ERR =" << res.stderror().transpose() << "\n";
    std::cerr << "EST.  TAU =" << res.tau().transpose() << "\n";

    // perform T2 test
    alps::alea::t2_result t2 = alps::alea::test_mean(res, model.mean());
    ASSERT_GE(t2.pvalue(), 0.01);
}

TEST(var1_test, same)
{
    Eigen::VectorXd phi0(2), veps_one(2), veps_two(2);
    Eigen::MatrixXd phi1(2,2);
    phi0 << 2, 3;
    phi1 << .90, 0, 0, .30;
    veps_one << 1.0, 0.25;
    veps_two << 2.0, 3.25;

    alps::alea::util::var1_model<double> model1(phi0, phi1, veps_one);
    alps::alea::util::var1_model<double> model2(phi0, phi1, veps_two);

    alps::alea::autocorr_acc<double> acc1(2), acc2(2);
    fill(model1, acc1, 400000);
    fill(model2, acc2, 400000);

    alps::alea::autocorr_result<double> res1 = acc1.finalize();
    alps::alea::autocorr_result<double> res2 = acc2.finalize();

    std::cerr << "EST.  MEAN=" << res1.mean().transpose() << "\n";
    std::cerr << "EST.  ERR =" << res1.stderror().transpose() << "\n";
    std::cerr << "EST.  MEAN=" << res2.mean().transpose() << "\n";
    std::cerr << "EST.  ERR =" << res2.stderror().transpose() << "\n";

    // perform T2 test
    alps::alea::t2_result t2 = alps::alea::test_mean(res1, res2);
    ASSERT_GE(t2.pvalue(), 0.01);
}
