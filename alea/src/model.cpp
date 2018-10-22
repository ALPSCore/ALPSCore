#include <alps/alea/util/model.hpp>

namespace alps { namespace alea { namespace util {

template <typename T>
void var1_model<T>::init()
{
    if (phi1_.rows() != size() || phi1_.cols() != size())
        throw std::invalid_argument("Illegal size for phi");
    if (var_eps_.rows() != size())
        throw std::invalid_argument("Illegal size for var_eps");

    if ((var_eps_.array() < 0).any())
        throw std::invalid_argument("Negative variances in var_eps");

    // Otherwise the process is not stationary and the variance diverges
    Eigen::JacobiSVD<typename eigen<T>::matrix> phi1_svd(phi1_);
    if (phi1_svd.singularValues()[0] >= 1) {
        throw std::invalid_argument(
                "phi1 is not a contraction, largest singular value is " +
                std::to_string(phi1_svd.singularValues()[0]));
    }

    stddev_eps_ = var_eps_.cwiseSqrt();
}

template <typename T>
typename eigen<T>::col var1_model<T>::mean() const
{
    // stationarity implies that:
    //         E[X_t] = phi0 + phi1 * E[X_t] + E[eps_t]
    //                 = (1 - phi1)^-1 * phi0
    auto eye = eigen<T>::matrix::Identity(size(), size());
    return (eye - phi1_).colPivHouseholderQr().solve(phi0_);
}

template <typename T, typename DerivedA, typename DerivedB>
typename eigen<T>::matrix kronecker(const Eigen::MatrixBase<DerivedA> &a,
                                    const Eigen::MatrixBase<DerivedB> &b)
{
    typename eigen<T>::matrix res(a.rows() + b.rows(), a.cols() + b.cols());
    for (int j = 0; j != a.cols(); ++j)
        for (int i = 0; i != a.rows(); ++i)
            res.block(i*b.rows(), j*b.cols(), b.rows(), b.cols()) =  a(i,j)*b;

    return res;
}

template <typename T>
typename eigen<T>::matrix var1_model<T>::cov() const
{
    // stationarity implies that:
    //       Cov[X(t), X(t)] = phi1 * Cov[X(t), X(t)] * phi1^+ + Cov[eps(t), eps(t)]
    // from which it follows:
    //       vec Cov[X] = (1 - phi1 (x) phi1)^{-1} vec Cov[eps]
    //
    // TODO: the inversion scales as size()**6. However above is nothing but a
    //       discrete Lyapunov equation, so we should use the Bartels-Stewart
    //       algorithm instead.
    size_t size_flat = size() * size();
    typename eigen<T>::matrix phi1_kron = kronecker<T>(phi1_, phi1_);

    // flatten eps covariance
    typename eigen<T>::matrix epscov = var_eps_.asDiagonal();
    typename eigen<T>::const_col_map epscov_flat(epscov.data(), size_flat);

    // flatten result covariance
    typename eigen<T>::matrix result(size(), size());
    typename eigen<T>::col_map result_flat(result.data(), size_flat);

    // do computation
    auto flat_eye = eigen<T>::matrix::Identity(size_flat, size_flat);
    result_flat = (flat_eye - phi1_kron).colPivHouseholderQr().solve(epscov_flat);
    return result;
}

template <typename T>
typename eigen<typename make_real<T>::type>::col var1_model<T>::var() const
{
    return cov().diagonal().real();
}

template <typename T>
typename eigen<T>::matrix var1_model<T>::ctau() const
{
    // Since:
    //     C(t) = Cov[X(t0 + t), X(t0)] = phi * C(t-1) = phi^t Cov[X(t0)]
    // we find:
    //     tau = (1 - phi1)^{-1} * phi1
    auto eye = eigen<T>::matrix::Identity(size(), size());
    return (eye - phi1_).colPivHouseholderQr().solve(phi1_);
}

template class var1_model<double>;
template class var1_model<std::complex<double> >;


template <typename T>
var1_run<T>::var1_run()
    : model_(nullptr)
    , t_(0)
{ }

template <typename T>
var1_run<T>::var1_run(const var1_model<T> &model)
    : model_(&model)
    , t_(0)
    , xt_(model.phi0())
    , epst_(model.size())
{  }

template <typename T>
void var1_run<T>::restart()
{
    t_ = 0;
    xt_ = model_->phi0();
    epst_.fill(0);
}

template <typename T>
void var1_run<T>::update()
{
    assert(model_ != nullptr);
    ++t_;
    xt_ = model_->phi0() + model_->phi1() * xt_ + epst_;
}

template class var1_run<double>;
template class var1_run<std::complex<double> >;

}}}
