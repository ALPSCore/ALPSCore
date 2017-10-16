#include <alps/alea/covariance.hpp>
#include <alps/alea/internal/outer.hpp>

namespace alps { namespace alea {

template <typename T, typename Str>
cov_data<T,Str>::cov_data(size_t size)
    : data_(size)
    , data2_(size, size)
{
    reset();
}

template <typename T, typename Str>
void cov_data<T,Str>::reset()
{
    state_ = SUM;
    data_.fill(0);
    data2_.fill(0);
    count_ = 0;
}

template <typename T, typename Str>
void cov_data<T,Str>::unlock_mean() const
{
    if (state_ == SUM) {
        data_ /= count_;
        data2_ -= internal::outer<Str>(data_, data_);
        data2_ /= count_ - 1;
        state_ = MEAN;
    }
}

template <typename T, typename Str>
void cov_data<T,Str>::unlock_sum() const
{
    if (state_ == MEAN) {
        data2_ *= count_ - 1;
        data2_ += internal::outer<Str>(data_, data_);
        data_ *= count_;
        state_ = SUM;
    }
}

template class cov_data<double>;
template class cov_data<std::complex<double> >;
template class cov_data<std::complex<double>, elliptic_var<std::complex<double> > >;


template <typename T, typename Str>
cov_acc<T,Str>::cov_acc(size_t size, size_t bundle_size)
    : current_(size, bundle_size)
    , store_(size)
    , uplevel_(NULL)
{ }

template <typename T, typename Str>
void cov_acc<T,Str>::reset()
{
    current_.reset();
    store_.reset();
}

template <typename T, typename Str>
cov_acc<T,Str> &cov_acc<T,Str>::operator<<(computed<T> &source)
{
    source.add_to(sink<T>(current_.sum().data(), current_.size()));
    ++current_.count();

    if (current_.is_full())
        add_bundle();
    return *this;
}

template <typename T, typename Str>
void cov_acc<T,Str>::add_bundle()
{
    // add batch to average and squared
    current_.sum() /= current_.count();
    store_.unlock_sum();
    store_.data().noalias() += current_.sum();
    store_.data2().noalias() += internal::outer<Str>(current_.sum(), current_.sum());
    store_.count() += 1;

    // add batch mean also to uplevel
    if (uplevel_ != NULL)
        (*uplevel_) << current_.sum();

    current_.reset();
}

template <typename T, typename Str>
void cov_acc<T,Str>::get_var(sink<var_type> out) const
{
    typename eigen<var_type>::col_map out_map(out.data(), out.size());

    store_.unlock_mean();
    out_map.noalias() += store_.data2().diagonal().real();
}

template <typename T, typename Str>
void cov_acc<T,Str>::get_stderr(sink<var_type> out) const
{
    typename eigen<var_type>::col_map out_map(out.data(), out.size());

    store_.unlock_mean();
    out_map.noalias() += (store_.data2().diagonal().real() / store_.count()).cwiseSqrt();
}

template class cov_acc<double>;
template class cov_acc<std::complex<double> >;
template class cov_acc<std::complex<double>, elliptic_var<std::complex<double> > >;

}}
