#include <alps/alea/variance.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/serialize.hpp>

#include <alps/alea/internal/util.hpp>
#include <alps/alea/internal/format.hpp>

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
    // This also works for count_ == 0
    data_ /= count_;
    data2_ -= count_ * data_.cwiseAbs2();

    // In case of zero unbiased information, the variance is infinite.
    // However, data2_ is 0 in this case as well, so we need to handle it
    // specially to avoid 0/0 = nan while propagating intrinsic NaN's.
    const double nunbiased = count_ - count2_/count_;
    if (nunbiased == 0) {
        data2_ = data2_.array().isNaN().select(data2_, INFINITY);
    } else {
        // HACK: this is written in out-of-place notation to work around Eigen
        data2_ = data2_ / nunbiased;
    }
}

template <typename T, typename Str>
void var_data<T,Str>::convert_to_sum()
{
    // "empty" sets must be handled specially here because of NaNs
    if (count_ == 0) {
        reset();
        return;
    }

    // Care must be taken again for zero unbiased info since inf/0 is NaN.
    const double nunbiased = count_ - count2_/count_;
    if (nunbiased == 0)
        data2_ = data2_.array().isNaN().select(data2_, 0);
    else
        data2_ = data2_ * nunbiased;

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
var_acc<T,Str> &var_acc<T,Str>::operator<<(const var_result<T,Str> &other)
{
    internal::check_valid(*this);
    if (size() != other.size())
        throw size_mismatch();

    // NOTE partial sums are unchanged
    // HACK we need this for "outwardly constant" manipulation
    var_data<T,Str> &other_store = const_cast<var_data<T,Str> &>(other.store());
    other_store.convert_to_sum();
    store_->data() += other_store.data();
    store_->data2() += other_store.data2();
    store_->count() += other_store.count();
    store_->count2() += other_store.count2();
    other_store.convert_to_mean();
    return *this;
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

template <typename T, typename Strategy>
bool operator==(const var_result<T, Strategy> &r1, const var_result<T, Strategy> &r2)
{
    if (r1.count() == 0 && r2.count() == 0)
        return true;

    return r1.count() == r2.count()
        && r1.count2() == r2.count2()
        && r1.store().data() == r2.store().data()
        && r1.store().data2() == r2.store().data2();
}

template bool operator==(const var_result<double> &r1, const var_result<double> &r2);
template bool operator==(const var_result<std::complex<double>, circular_var> &r1,
                         const var_result<std::complex<double>, circular_var> &r2);
template bool operator==(const var_result<std::complex<double>, elliptic_var> &r1,
                         const var_result<std::complex<double>, elliptic_var> &r2);

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
    internal::serializer_sentry group(s, key);

    serialize(s, "@size", self.store_->data_.size());
    serialize(s, "count", self.store_->count_);
    serialize(s, "count2", self.store_->count2_);
    s.enter("mean");
    serialize(s, "value", self.store_->data_);
    serialize(s, "error", self.stderror());   // TODO temporary
    s.exit();
    serialize(s, "var", self.store_->data2_);
}

template <typename T, typename Str>
void deserialize(deserializer &s, const std::string &key, var_result<T,Str> &self)
{
    typedef typename var_result<T,Str>::var_type var_type;
    internal::deserializer_sentry group(s, key);

    // first deserialize the fundamentals and make sure that the target fits
    size_t new_size;
    deserialize(s, "@size", new_size);
    if (!self.valid() || self.size() != new_size)
        self.store_.reset(new var_data<T,Str>(new_size));

    // deserialize data
    deserialize(s, "count", self.store_->count_);
    deserialize(s, "count2", self.store_->count2_);
    s.enter("mean");
    deserialize(s, "value", self.store_->data_);
    s.read("error", ndview<var_type>(nullptr, &new_size, 1)); // discard
    s.exit();
    deserialize(s, "var", self.store_->data2_);
}

template void serialize(serializer &, const std::string &key, const var_result<double, circular_var> &);
template void serialize(serializer &, const std::string &key, const var_result<std::complex<double>, circular_var> &);
template void serialize(serializer &, const std::string &key, const var_result<std::complex<double>, elliptic_var> &);

template void deserialize(deserializer &, const std::string &key, var_result<double, circular_var> &);
template void deserialize(deserializer &, const std::string &key, var_result<std::complex<double>, circular_var> &);
template void deserialize(deserializer &, const std::string &key, var_result<std::complex<double>, elliptic_var> &);


template <typename T, typename Str>
std::ostream &operator<<(std::ostream &str, const var_result<T,Str> &self)
{
    internal::check_valid(self);
    internal::format_sentry sentry(str);
    verbosity verb = internal::get_format(str, PRINT_TERSE);

    if (verb == PRINT_VERBOSE)
        str << "<X> = ";
    str << self.mean() << " +- " << self.stderror();
    return str;
}

template std::ostream &operator<<(std::ostream &, const var_result<double, circular_var> &);
template std::ostream &operator<<(std::ostream &, const var_result<std::complex<double>, circular_var> &);
template std::ostream &operator<<(std::ostream &, const var_result<std::complex<double>, elliptic_var> &);

}}
