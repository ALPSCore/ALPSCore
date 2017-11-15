#include <alps/alea/variance.hpp>
#include <alps/alea/util.hpp>

#include <alps/alea/internal/util.hpp>

namespace alps { namespace alea {

template <typename T, typename Str>
var_data<T,Str>::var_data(size_t size)
    : data_(size)
    , data2_(size)
{
    reset();
}

template <typename T, typename Str>
void var_data<T,Str>::reset()
{
    data_.fill(0);
    data2_.fill(0);
    count_ = 0;
}

template <typename T, typename Str>
void var_data<T,Str>::convert_to_mean()
{
    data_ /= count_;
    data2_ -= data_.cwiseAbs2();
    data2_ /= count_ - 1;
}

template <typename T, typename Str>
void var_data<T,Str>::convert_to_sum()
{
    data2_ *= count_ - 1;
    data2_ += data_.cwiseAbs2();
    data_ *= count_;
}

template class var_data<double>;
template class var_data<std::complex<double> >;
template class var_data<std::complex<double>, elliptic_var<std::complex<double> > >;


template <typename T, typename Str>
var_acc<T,Str>::var_acc()
    : store_()
    , current_(0, 1)
    , uplevel_(NULL)
    , initialized_(false)
{ }

template <typename T, typename Str>
var_acc<T,Str>::var_acc(size_t size, size_t bundle_size)
    : store_(new var_data<T,Str>(size))
    , current_(size, bundle_size)
    , uplevel_(NULL)
    , initialized_(true)
{ }

// We need an explicit copy constructor, as we need to copy the data
template <typename T, typename Str>
var_acc<T,Str>::var_acc(const var_acc &other)
    : store_(other.store_ ? new var_data<T,Str>(*other.store_) : NULL)
    , current_(other.current_)
    , uplevel_(other.uplevel_)
    , initialized_(other.initialized_)
{ }

template <typename T, typename Str>
var_acc<T,Str> &var_acc<T,Str>::operator=(const var_acc &other)
{
    store_.reset(other.store_ ? new var_data<T,Str>(*other.store_) : NULL);
    current_ = other.current_;
    uplevel_ = other.uplevel_;
    initialized_ = other.initialized_;
    return *this;
}

template <typename T, typename Str>
void var_acc<T,Str>::reset()
{
    internal::check_init(*this);
    current_.reset();
    if (valid())
        store_->reset();
    else
        store_.reset(new var_data<T,Str>(size()));
}

template <typename T, typename Str>
var_acc<T,Str> &var_acc<T,Str>::operator<<(const computed<value_type> &source)
{
    internal::check_valid(*this);
    source.add_to(sink<T>(current_.sum().data(), current_.size()));
    ++current_.count();

    if (current_.is_full())
        add_bundle();
    return *this;
}

template <typename T, typename Str>
var_result<T,Str> var_acc<T,Str>::result() const
{
    internal::check_valid(*this);
    var_result<T,Str> result(*store_);
    result.store_->convert_to_mean();
    return result;
}

template <typename T, typename Str>
var_result<T,Str> var_acc<T,Str>::finalize()
{
    internal::check_valid(*this);
    var_result<T,Str> result(*store_);
    result.store_.swap(store_);
    result.store_->convert_to_mean();
    return result;
}

template <typename T, typename Str>
void var_acc<T,Str>::add_bundle()
{
    typename Str::abs2_op abs2;

    // add batch to average and squared
    current_.sum() /= current_.count();
    store_->data().noalias() += current_.sum();
    store_->data2().noalias() += current_.sum().unaryExpr(abs2);
    store_->count() += 1;

    // add batch mean also to uplevel
    if (uplevel_ != NULL)
        (*uplevel_) << current_.sum();

    current_.reset();
}

template class var_acc<double>;
template class var_acc<std::complex<double> >;
template class var_acc<std::complex<double>, elliptic_var<std::complex<double> > >;


template <typename T, typename Str>
column<typename var_result<T,Str>::var_type> var_result<T,Str>::stderror() const
{
    throw invalid_accumulator();
}

template class var_result<double>;
template class var_result<std::complex<double> >;
template class var_result<std::complex<double>, elliptic_var<std::complex<double> > >;

}}
