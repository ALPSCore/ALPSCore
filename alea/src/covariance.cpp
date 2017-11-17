#include <alps/alea/covariance.hpp>
#include <alps/alea/internal/outer.hpp>
#include <alps/alea/internal/util.hpp>

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
    data_.fill(0);
    data2_.fill(0);
    count_ = 0;
}

template <typename T, typename Str>
void cov_data<T,Str>::convert_to_mean()
{
    data_ /= count_;
    data2_ -= count_ * internal::outer<Str>(data_, data_);
    data2_ /= count_ - 1;
}

template <typename T, typename Str>
void cov_data<T,Str>::convert_to_sum()
{
    data2_ *= count_ - 1;
    data2_ += count_ * internal::outer<Str>(data_, data_);
    data_ *= count_;
}

template class cov_data<double>;
template class cov_data<std::complex<double> >;
template class cov_data<std::complex<double>, elliptic_var<std::complex<double> > >;


template <typename T, typename Str>
cov_acc<T,Str>::cov_acc()
    : store_()
    , current_(0, 1)
    , uplevel_(NULL)
    , initialized_(false)
{ }

template <typename T, typename Str>
cov_acc<T,Str>::cov_acc(size_t size, size_t bundle_size)
    : store_(new cov_data<T,Str>(size))
    , current_(size, bundle_size)
    , uplevel_(NULL)
    , initialized_(true)
{ }

// We need an explicit copy constructor, as we need to copy the data
template <typename T, typename Str>
cov_acc<T,Str>::cov_acc(const cov_acc &other)
    : store_(other.store_ ? new cov_data<T,Str>(*other.store_) : NULL)
    , current_(other.current_)
    , uplevel_(other.uplevel_)
    , initialized_(other.initialized_)
{ }

template <typename T, typename Str>
cov_acc<T,Str> &cov_acc<T,Str>::operator=(const cov_acc &other)
{
    store_.reset(other.store_ ? new cov_data<T,Str>(*other.store_) : NULL);
    current_ = other.current_;
    uplevel_ = other.uplevel_;
    initialized_ = other.initialized_;
    return *this;
}

template <typename T, typename Str>
void cov_acc<T,Str>::reset()
{
    internal::check_init(*this);
    current_.reset();
    if (valid())
        store_->reset();
    else
        store_.reset(new cov_data<T,Str>(size()));
}

template <typename T, typename Str>
cov_acc<T,Str> &cov_acc<T,Str>::operator<<(const computed<value_type> &source)
{
    internal::check_valid(*this);
    source.add_to(sink<T>(current_.sum().data(), current_.size()));
    ++current_.count();

    if (current_.is_full())
        add_bundle();
    return *this;
}

template <typename T, typename Str>
cov_result<T,Str> cov_acc<T,Str>::result() const
{
    internal::check_valid(*this);
    cov_result<T,Str> result(*store_);
    result.store_->convert_to_mean();
    return result;
}

template <typename T, typename Str>
cov_result<T,Str> cov_acc<T,Str>::finalize()
{
    cov_result<T,Str> result;
    finalize_to(result);
    return result;
}

template <typename T, typename Str>
void cov_acc<T,Str>::finalize_to(cov_result<T,Str> &result)
{
    internal::check_valid(*this);
    if (result.valid())
        throw std::runtime_error("Can only finalize to uninitialized result");

    result.store_.swap(store_);
    result.store_->convert_to_mean();
}

template <typename T, typename Str>
void cov_acc<T,Str>::add_bundle()
{
    // add batch to average and squared
    current_.sum() /= current_.count();
    store_->data().noalias() += current_.sum();
    store_->data2().noalias() += internal::outer<Str>(current_.sum(), current_.sum());
    store_->count() += 1;

    // add batch mean also to uplevel
    if (uplevel_ != NULL)
        (*uplevel_) << current_.sum();

    current_.reset();
}

template class cov_acc<double>;
template class cov_acc<std::complex<double> >;
template class cov_acc<std::complex<double>, elliptic_var<std::complex<double> > >;


template <typename T, typename Str>
column<typename cov_result<T,Str>::var_type> cov_result<T,Str>::stderror() const
{
    internal::check_valid(*this);
    return (store_->data2().diagonal().real() / store_->count()).cwiseSqrt();
}

template <typename T, typename Str>
void cov_result<T,Str>::reduce(reducer &r)
{
    internal::check_valid(*this);
    store_->convert_to_sum();
    reducer_setup setup = r.get_setup();
    r.reduce(sink<T>(store_->data().data(), store_->data().rows()));
    r.reduce(sink<cov_type>(store_->data2().data(), store_->data2().size()));
    r.reduce(sink<size_t>(&store_->count(), 1));
    r.commit();

    if (setup.have_result)
        store_->convert_to_mean();
    else
        store_.reset();   // free data
}

template class cov_result<double>;
template class cov_result<std::complex<double> >;
template class cov_result<std::complex<double>, elliptic_var<std::complex<double> > >;

}}
