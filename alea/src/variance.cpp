#include <alps/alea/variance.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/serialize.hpp>

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
    count2_ = 0;
}

template <typename T, typename Str>
void var_data<T,Str>::convert_to_mean()
{
    data_ /= count_;
    data2_ -= count_ * data_.cwiseAbs2();

    // bias correction: count2_/count_ is 1 for (non-weighted) mean
    data2_ = data2_ / (count_ - count2_/count_);
}

template <typename T, typename Str>
void var_data<T,Str>::convert_to_sum()
{
    data2_ = data2_ * (count_ - count2_/count_);
    data2_ += count_ * data_.cwiseAbs2();
    data_ *= count_;
}

template class var_data<double>;
template class var_data<std::complex<double>, circular_var>;
template class var_data<std::complex<double>, elliptic_var>;


template <typename T, typename Str>
var_acc<T,Str>::var_acc(size_t size, size_t bundle_size)
    : store_(new var_data<T,Str>(size))
    , current_(size, bundle_size)
{ }

// We need an explicit copy constructor, as we need to copy the data
template <typename T, typename Str>
var_acc<T,Str>::var_acc(const var_acc &other)
    : store_(other.store_ ? new var_data<T,Str>(*other.store_) : nullptr)
    , current_(other.current_)
{ }

template <typename T, typename Str>
var_acc<T,Str> &var_acc<T,Str>::operator=(const var_acc &other)
{
    store_.reset(other.store_ ? new var_data<T,Str>(*other.store_) : nullptr);
    current_ = other.current_;
    return *this;
}

template <typename T, typename Str>
void var_acc<T,Str>::reset()
{
    current_.reset();
    if (valid())
        store_->reset();
    else
        store_.reset(new var_data<T,Str>(size()));
}

template <typename T, typename Str>
void var_acc<T,Str>::add(const computed<T> &source, size_t count,
                         var_acc<T,Str> *cascade)
{
    internal::check_valid(*this);
    source.add_to(view<T>(current_.sum().data(), current_.size()));
    current_.count() += count;

    if (current_.is_full())
        add_bundle(cascade);
}

template <typename T, typename Str>
var_result<T,Str> var_acc<T,Str>::result() const
{
    internal::check_valid(*this);
    var_result<T,Str> result;
    var_acc<T,Str>(*this).finalize_to(result, nullptr);
    return result;
}

template <typename T, typename Str>
var_result<T,Str> var_acc<T,Str>::finalize()
{
    var_result<T,Str> result;
    finalize_to(result, nullptr);
    return result;
}

template <typename T, typename Str>
void var_acc<T,Str>::finalize_to(var_result<T,Str> &result, var_acc<T,Str> *cascade)
{
    internal::check_valid(*this);

    // add leftover data to the variance.  The upwards propagation must be
    // handled by going through a hierarchy in ascending order.
    if (current_.count() != 0)
        add_bundle(cascade);

    // data swap
    result.store_.reset();
    result.store_.swap(store_);

    // post-processing to result
    result.store_->convert_to_mean();
}

template <typename T, typename Str>
void var_acc<T,Str>::add_bundle(var_acc<T,Str> *cascade)
{
    typename bind<Str, T>::abs2_op abs2;

    // add batch to average and squared
    store_->data().noalias() += current_.sum();
    store_->data2().noalias() += current_.sum().unaryExpr(abs2) / current_.count();
    store_->count() += current_.count();
    store_->count2() += current_.count() * current_.count();

    // add batch mean also to uplevel
    if (cascade != nullptr)
        cascade->add(make_adapter(current_.sum()), current_.count(), cascade+1);

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
    return (store_->data2() / observations()).cwiseSqrt();
}

template <typename T, typename Str>
void var_result<T,Str>::reduce(const reducer &r, bool pre_commit, bool post_commit)
{
    internal::check_valid(*this);
    if (pre_commit) {
        store_->convert_to_sum();
        r.reduce(view<T>(store_->data().data(), store_->data().rows()));
        r.reduce(view<var_type>(store_->data2().data(), store_->data2().rows()));
        r.reduce(view<double>(&store_->count(), 1));
        r.reduce(view<double>(&store_->count2(), 1));
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


template <typename T, typename Str>
void serialize(serializer &s, const std::string &key, const var_result<T,Str> &self)
{
    internal::check_valid(self);
    internal::group_sentry group(s, key);

    serialize(s, "@size", self.store_->data_.size());
    serialize(s, "count", self.store_->count_);
    serialize(s, "count2", self.store_->count2_);
    s.enter("mean");
    serialize(s, "value", self.store_->data_);
    serialize(s, "error", self.stderror());   // TODO temporary
    s.exit();
    serialize(s, "var", self.store_->data2_);
}

template void serialize(serializer &, const std::string &key, const var_result<double, circular_var> &);
template void serialize(serializer &, const std::string &key, const var_result<std::complex<double>, circular_var> &);
template void serialize(serializer &, const std::string &key, const var_result<std::complex<double>, elliptic_var> &);

}}
