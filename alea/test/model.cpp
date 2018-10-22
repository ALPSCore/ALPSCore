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

void print_t2(std::ostream &out, const alps::alea::t2_result &res)
{
    out << "T2 RESULT:"
        << "\n\ta (size) = " << res.dist().degrees_of_freedom1()
        << "\n\tb (dof)  = " << res.dist().degrees_of_freedom2()
        << "\n\tFab   1% = " << quantile(res.dist(), 0.01)
        << "\n\tFab   5% = " << quantile(res.dist(), 0.05)
        << "\n\tFab  25% = " << quantile(res.dist(), 0.25)
        << "\n\tFab  50% = " << quantile(res.dist(), 0.50)
        << "\n\tFab  75% = " << quantile(res.dist(), 0.75)
        << "\n\tFab  95% = " << quantile(res.dist(), 0.95)
        << "\n\tFab  99% = " << quantile(res.dist(), 0.99)
        << "\n\tf-score  = " << res.score()
        << "\n\tp-values = " << res.pvalue_lower() << " " << res.pvalue_upper()
        << std::endl;
}

template <typename T>
void print_result(std::ostream &out, const alps::alea::autocorr_result<T> &res)
{
    out << "ESTIMATED RESULT:"
        << "\n\tmean = " << res.mean().transpose()
        << "\n\tsem  = " << res.stderror().transpose()
        << "\n\tvar  = " << res.var().transpose()
        << "\n\tnobs = " << res.observations()
        << "\n\ttau  = " << res.tau().transpose()
        << std::endl;
}

template <typename T>
void print_result(std::ostream &out, const alps::alea::var_result<T> &res)
{
    out << "ESTIMATED RESULT:"
        << "\n\tmean = " << res.mean().transpose()
        << "\n\tsem  = " << res.stderror().transpose()
        << "\n\tvar  = " << res.var().transpose()
        << "\n\tnobs = " << res.observations()
        << std::endl;
}

template <typename Acc>
class model_error_case
    : public ::testing::Test
{
public:
    typedef typename alps::alea::traits<Acc>::var_type var_type;

    model_error_case()
        : acc_(2)
        , model_()
    {
        Eigen::VectorXd phi0(2), veps(2);
        Eigen::MatrixXd phi1(2,2);

        phi0 << 2, 3;
        phi1 << .80, 0, 0, .64;
        veps << 1.0, 0.25;
        model_ = alps::alea::util::var1_model<double>(phi0, phi1, veps);
    }

    void test()
    {
        acc_.set_batch_size(4);
        fill(model_, acc_, 400000);
        result_ = acc_.finalize();

        std::vector<double> obs_var = result_.var();
        std::vector<double> obs_stderr = result_.stderror();
        double nobs = result_.observations();

        EXPECT_NEAR(result_.count() * result_.count() / result_.count2(),
                    nobs, 1e-10);
        EXPECT_NEAR(std::sqrt(obs_var[0]/nobs), obs_stderr[0], 1e-10);
        EXPECT_NEAR(std::sqrt(obs_var[0]/nobs), obs_stderr[0], 1e-10);
    }

private:
    Acc acc_;
    typename alps::alea::traits<Acc>::result_type result_;
    alps::alea::util::var1_model<double> model_;
};

typedef ::testing::Types<
      alps::alea::var_acc<double>
    , alps::alea::cov_acc<double>
    , alps::alea::autocorr_acc<double>
    , alps::alea::batch_acc<double>
    > has_stderr;

TYPED_TEST_CASE(model_error_case, has_stderr);
TYPED_TEST(model_error_case, test) { this->test(); }


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
    print_result(std::cerr, res);

    // perform T2 test
    alps::alea::t2_result t2 = alps::alea::test_mean(res, model.mean());
    print_t2(std::cerr, t2);
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

    std::cerr << "EXACT MEAN=" << model1.mean().transpose() << "\n";

    alps::alea::autocorr_acc<double> acc1(2), acc2(2);
    fill(model1, acc1, 400000);
    fill(model2, acc2, 400000);

    alps::alea::autocorr_result<double> res1 = acc1.finalize();
    print_result(std::cerr, res1);

    alps::alea::autocorr_result<double> res2 = acc2.finalize();
    print_result(std::cerr, res2);

    // perform T2 test manually
    alps::alea::var_result<double> diff = alps::alea::internal::pool_var(res1, res2);
    print_result(std::cerr, diff);
    alps::alea::t2_result t2 = alps::alea::t2_test(diff.mean(), diff.var(),
                                                   diff.observations(), 1, 1e-10);
    print_t2(std::cerr, t2);
    ASSERT_GE(t2.pvalue(), 0.01);

    // Perform T2 test automatically
    t2 = alps::alea::test_mean(res1, res2);
    ASSERT_GE(t2.pvalue(), 0.01);
}
