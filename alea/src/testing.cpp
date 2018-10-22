#include <alps/alea/testing.hpp>

namespace alps { namespace alea {

template <typename T>
t2_result t2_test(const column<T> &diff,
                  const column<typename make_real<T>::type> &var,
                  double nmeas, size_t pools, double atol)
{
    if (diff.rows() != var.rows())
        throw std::invalid_argument("Size mismatch between diff and var");
    if (pools != 1 && pools != 2)
        throw std::invalid_argument("Pools must be 1 or 2");
    if (nmeas <= 0)
        throw std::invalid_argument("Must have at least 1 measurement");
    if ((var.array() < -atol).any())
        throw std::invalid_argument("Variances must be positive");

    // remove data points where both the variance and difference are zero
    // within atol while propagating intrinsic NaNs
    double t2 = 0;
    size_t nsize = 0;
    for (int i = 0; i != diff.rows(); ++i) {
        if (std::abs(var[i]) < atol && std::norm(diff[i]) < atol)
            continue;

        t2 += std::norm(diff[i]) / var[i];
        ++nsize;
    }
    t2 *= nmeas;

    // compute degrees of freedom.
    double dof = nmeas - nsize;
    if (dof <= pools)
        return t2_result(NAN, nsize, dof);

    // normalize T2 score to match F distribution
    double score = dof/(nsize * (nmeas - pools)) * t2;

    return t2_result(score, nsize, dof);
}

template t2_result t2_test(const column<double> &diff, const column<double> &var,
                           double nmeas, size_t pools, double atol);
template t2_result t2_test(const column<std::complex<double>> &diff,
                           const column<double> &var, double nmeas, size_t pools,
                           double atol);

}} /* namespace alps::alea */

