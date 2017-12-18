#include <alps/alea/variance.hpp>
#include <alps/alea/util.hpp>

#include <alps/alea/internal/util.hpp>

namespace alps { namespace alea {

template <typename T, typename Str>
var_data<T,Str>::var_data(size_t size, size_t batch_size)
    : data_(size)
    , data2_(size)
    , batch_size_(batch_size)
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
    data2_ -= count_ * data_.cwiseAbs2();
    data2_ /= count_ - 1;
}

template <typename T, typename Str>
void var_data<T,Str>::convert_to_sum()
{
    data2_ *= count_ - 1;
    data2_ += count_ * data_.cwiseAbs2();
    data_ *= count_;
}

template class var_data<double>;
template class var_data<std::complex<double>, circular_var>;
template class var_data<std::complex<double>, elliptic_var>;


template <typename T, typename Str>
var_acc<T,Str>::var_acc(size_t size, size_t bundle_size)
    : store_(new var_data<T,Str>(size, bundle_size))
    , current_(size, bundle_size)
    , uplevel_(nullptr)
{ }

// We need an explicit copy constructor, as we need to copy the data
template <typename T, typename Str>
var_acc<T,Str>::var_acc(const var_acc &other)
    : store_(other.store_ ? new var_data<T,Str>(*other.store_) : nullptr)
    , current_(other.current_)
    , uplevel_(other.uplevel_)
{ }

template <typename T, typename Str>
var_acc<T,Str> &var_acc<T,Str>::operator=(const var_acc &other)
{
    store_.reset(other.store_ ? new var_data<T,Str>(*other.store_) : nullptr);
    current_ = other.current_;
    uplevel_ = other.uplevel_;
    return *this;
}

template <typename T, typename Str>
void var_acc<T,Str>::reset()
{
    current_.reset();
    if (valid())
        store_->reset();
    else
        store_.reset(new var_data<T,Str>(size(), batch_size()));
}

template <typename T, typename Str>
void var_acc<T,Str>::add(const computed<T> &source, size_t count)
{
    internal::check_valid(*this);
    source.add_to(sink<T>(current_.sum().data(), current_.size()));
    current_.count() += count;

    if (current_.is_full())
        add_bundle();
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
    var_result<T,Str> result;
    finalize_to(result);
    return result;
}

template <typename T, typename Str>
void var_acc<T,Str>::finalize_to(var_result<T,Str> &result)
{
    internal::check_valid(*this);
    result.store_.reset();
    result.store_.swap(store_);
    result.store_->convert_to_mean();
}

template <typename T, typename Str>
void var_acc<T,Str>::add_bundle()
{
    typename bind<Str, T>::abs2_op abs2;

    // add batch to average and squared
    current_.sum() /= current_.count();
    store_->data().noalias() += current_.sum();
    store_->data2().noalias() += current_.sum().unaryExpr(abs2);
    store_->count() += 1;

    // add batch mean also to uplevel
    if (uplevel_ != nullptr)
        (*uplevel_) << current_.sum();

    current_.reset();
}

template class var_acc<double>;
template class var_acc<std::complex<double>, circular_var>;
template class var_acc<std::complex<double>, elliptic_var>;

// We need an explicit copy constructor, as we need to copy the data
template <typename T, typename Str>
var_result<T,Str>::var_result(const var_result &other)
    : store_(other.store_ ? new var_data<T,Str>(*other.store_) : nullptr)
{ }

template <typename T, typename Str>
var_result<T,Str> &var_result<T,Str>::operator=(const var_result &other)
{
    store_.reset(other.store_ ? new var_data<T,Str>(*other.store_) : nullptr);
    return *this;
}

template <typename T, typename Str>
column<typename var_result<T,Str>::var_type> var_result<T,Str>::stderror() const
{
    internal::check_valid(*this);
    return (store_->data2() / store_->count()).cwiseSqrt();
}

template <typename T, typename Str>
void var_result<T,Str>::reduce(const reducer &r, bool pre_commit, bool post_commit)
{
    internal::check_valid(*this);
    if (pre_commit) {
        store_->convert_to_sum();
        r.reduce(sink<T>(store_->data().data(), store_->data().rows()));
        r.reduce(sink<var_type>(store_->data2().data(), store_->data2().rows()));
        r.reduce(sink<size_t>(&store_->count(), 1));
    }
    if (pre_commit && post_commit) {
        r.commit();
    }
    if (post_commit) {
        reducer_setup setup = r.get_setup();
        if (setup.have_result)
            store_->convert_to_mean();
        else
            store_.reset();   // free data
    }
}


template class var_result<double>;
template class var_result<std::complex<double>, circular_var>;
template class var_result<std::complex<double>, elliptic_var>;

}}
