#include <random>
#include <iostream>

#include <alps/alea.hpp>
#include <alps/alea/util/model.hpp>

#include <Eigen/Dense>

using namespace alps;

alea::util::var1_model<double> create_model()
{
    Eigen::VectorXd phi0(1), eps(1);
    Eigen::MatrixXd phi1(1,1);

    phi0 << 3.0;
    phi1 << 0.97;
    eps << 2.0;
    return alea::util::var1_model<double>(phi0, phi1, eps);
}

int main()
{
    // Construct a data set from a very simple prescription called VAR(1)
    alea::util::var1_model<double> model = create_model();
    std::cout << "Exact <X> =" << model.mean().transpose() << "\n";
    std::cout << "Exact autocorr. time = " << model.ctau() << "\n\n";

    // Set up two accumulators: one which tries to estimate autocorrelation
    // time and one which keeps track of the distribution of values.  The "1"
    // denotes the number of vector components (1).
    alea::autocorr_acc<double> acorr(1);
    alea::batch_acc<double> abatch(1);

    // Set up random number generator
    std::mt19937 prng(0);

    // Generate data points and add them to the accumulators
    std::cout << "RUNNING SIMULATION\n";
    alea::util::var1_run<double> generator = model.start();
    while(generator.t() < 1000000) {
        // Measure the current value
        double current = generator.xt()[0];

        // Add current data point to the accumulator.  Both accumulators
        // have very little runtime overhead.
        acorr << current;
        abatch << current;

        // Perform step
        generator.step(prng);
    }

    // Analyze data: we repurpose our accumulated data as results by calling
    // finalize.  Before we can accumulate again, we would need to call reset()
    // on the accumulators
    alea::autocorr_result<double> rcorr = acorr.finalize();
    alea::batch_result<double> rbatch = abatch.finalize();

    // The autocorrelation accumulator measures the autocorrelation time and
    // corrects the standard error of the mean accordingly.
    std::cout << "Measured <X> = " << rcorr << "\n";
    std::cout << "Measured autocorr. time = " << rcorr.tau() << "\n";

    return 0;
}
