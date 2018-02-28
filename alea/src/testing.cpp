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
        if (abs(var[i]) < atol && std::norm(diff[i]) < atol)
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

template <typename T>
diag_diffs<T> diagonalize_cov(const column<T> &diff,
                              const typename eigen<T>::matrix &cov)
{
    if (!cov.isApprox(cov.adjoint()))
        throw std::invalid_argument("Covariance matrix is not Hermitean");

    Eigen::SelfAdjointEigenSolver<typename eigen<T>::matrix> eigen(cov);
    return diag_diffs<T>{
                eigen.eigenvectors().adjoint() * diff,
                eigen.eigenvalues()
                };
}

template diag_diffs<double> diagonalize_cov(const column<double> &diff,
                                const typename eigen<double>::matrix &cov);
template diag_diffs<std::complex<double>> diagonalize_cov(
                                const column<std::complex<double>> &diff,
                                const typename eigen<std::complex<double>>::matrix &cov);


}} /* namespace alps::alea */

