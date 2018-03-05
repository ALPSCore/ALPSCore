#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>

#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>

#include <alps/alea/internal/joined.hpp>

namespace alps { namespace alea { namespace internal {

template <typename T, typename Str>
const typename eigen<typename traits<cov_result<T,Str>>::cov_type>::matrix &
get_cov(const cov_result<T,Str> &result)
{
    return result.cov();
}

template <typename Result>
typename eigen<typename traits<Result>::cov_type>::matrix
get_cov(const Result &result)
{
    return result.cov();
}

template <typename T, typename Str>
typename eigen<typename traits<cov_result<T,Str>>::cov_type>::matrix
get_cov(const var_result<T,Str> &result)
{
    return result.var().asDiagonal();
}

/**
 * Get difference
 */
template <typename Result, typename ExpectedT,
          typename T=add_scalar_type<typename traits<Result>::value_type, ExpectedT>>
var_result<T> make_diff(const Result &result, const column<ExpectedT> &expected)
{
    if (result.size() != expected.size())
        throw size_mismatch();

    var_result<T> diff(var_data<T>(result.size()));
    diff.store().count() = result.count();
    diff.store().count2() = result.count2();
    diff.store().data() = result.mean() - expected;

    if (traits<Result>::HAVE_COV) {
        Eigen::SelfAdjointEigenSolver<typename eigen<T>::matrix> ecov(result.cov());
        diff.store().data() = ecov.eigenvectors().adjoint() * diff.store().data();
        diff.store().data2() = ecov.eigenvalues();
    }
    return diff;
}

/**
 * Return result with pooled variance
 */
template <typename Result1, typename Result2,
          typename T=joined_value_type<Result1, Result2>>
var_result<T> pool_var(const Result1 &r1, const Result2 &r2)
{
    if (r1.size() != r2.size())
        throw size_mismatch();

    var_result<T> pooled(var_data<T>(r1.size()));
    pooled.store().count() = r1.count() * r2.count() / (r1.count() + r2.count());
    pooled.store().count2() = 0;   // FIXME (does not matter for t2 test)
    pooled.store().data() = r1.mean() - r2.mean();

    if (traits<Result1>::HAVE_COV || traits<Result2>::HAVE_COV) {
        // Pooling covariance matrices - diagonalize those to yield variances
        auto pooled_cov = r1.count() * get_cov(r1) + r2.count() * get_cov(r2)
                          / (r1.count() + r2.count() - 2.0);

        Eigen::SelfAdjointEigenSolver<typename eigen<T>::matrix> ecov(pooled_cov);
        pooled.store().data() = ecov.eigenvectors().adjoint() * pooled.store().data();
        pooled.store().data2() = ecov.eigenvalues();
    } else {
        // Directly pooling variances
        pooled.store().data2() = r1.count() * r1.var() + r2.count() * r2.var()
                                / (r1.count() + r2.count() - 2.0);
    }
    return pooled;
}

}}} /* namespace alps::alea::internal */
