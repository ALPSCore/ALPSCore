/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/alea/covariance.hpp>
#include <alps/alea/serialize.hpp>

#include <alps/alea/internal/outer.hpp>
#include <alps/alea/internal/util.hpp>
#include <alps/alea/internal/format.hpp>

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
    count2_ = 0;
}

template <typename T, typename Str>
void cov_data<T,Str>::convert_to_mean()
{
    data_ /= count_;
    data2_ -= count_ * internal::outer<bind<Str, T> >(data_, data_);

    // In case of zero unbiased information, the variance is infinite.
    // However, data2_ is 0 in this case as well, so we need to handle it
    // specially to avoid 0/0 = nan while propagating intrinsic NaN's.
    const double nunbiased = count_ - count2_/count_;
    if (nunbiased == 0)
        data2_ = data2_.array().isNaN().select(data2_, INFINITY);
    else
        // HACK: this is written in out-of-place notation to work around Eigen
        data2_ = data2_ / nunbiased;
}

template <typename T, typename Str>
void cov_data<T,Str>::convert_to_sum()
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

    data2_ += count_ * internal::outer<bind<Str, T> >(data_, data_);
    data_ *= count_;
}

template class cov_data<double>;
template class cov_data<std::complex<double>, circular_var>;
template class cov_data<std::complex<double>, elliptic_var>;


template <typename T, typename Str>
cov_acc<T,Str>::cov_acc(size_t size, size_t batch_size)
    : store_(new cov_data<T,Str>(size))
    , current_(size, batch_size)
{ }

// We need an explicit copy constructor, as we need to copy the data
template <typename T, typename Str>
cov_acc<T,Str>::cov_acc(const cov_acc &other)
    : store_(other.store_ ? new cov_data<T,Str>(*other.store_) : nullptr)
    , current_(other.current_)
{ }

template <typename T, typename Str>
cov_acc<T,Str> &cov_acc<T,Str>::operator=(const cov_acc &other)
{
    store_.reset(other.store_ ? new cov_data<T,Str>(*other.store_) : nullptr);
    current_ = other.current_;
    return *this;
}

template <typename T, typename Str>
void cov_acc<T,Str>::reset()
{
    current_.reset();
    if (valid())
        store_->reset();
    else
        store_.reset(new cov_data<T,Str>(size()));
}

template <typename T, typename Str>
void cov_acc<T,Str>::set_size(size_t size)
{
    current_ = bundle<T>(size, current_.target());
    if (valid())
        store_.reset(new cov_data<T,Str>(size));
}

template <typename T, typename Str>
void cov_acc<T,Str>::set_batch_size(size_t batch_size)
{
    // TODO: allow resizing with reset
    current_.target() = batch_size;
    current_.reset();
}

template <typename T, typename Str>
void cov_acc<T,Str>::add(const computed<value_type> &source, size_t count)
{
    internal::check_valid(*this);
    source.add_to(view<T>(current_.sum().data(), current_.size()));
    current_.count() += count;

    if (current_.is_full())
        add_bundle();
}

template <typename T, typename Str>
cov_acc<T,Str> &cov_acc<T,Str>::operator<<(const cov_result<T,Str> &other)
{
    internal::check_valid(*this);
    if (size() != other.size())
        throw size_mismatch();

    // NOTE partial sums are unchanged
    // HACK we need this for "outwardly constant" manipulation
    cov_data<T,Str> &other_store = const_cast<cov_data<T,Str> &>(other.store());
    other_store.convert_to_sum();
    store_->data() += other_store.data();
    store_->data2() += other_store.data2();
    store_->count() += other_store.count();
    store_->count2() += other_store.count2();
    other_store.convert_to_mean();
    return *this;
}

template <typename T, typename Str>
cov_result<T,Str> cov_acc<T,Str>::result() const
{
    internal::check_valid(*this);
    cov_result<T,Str> result(*store_);
    cov_acc<T,Str>(*this).finalize_to(result);
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

    // add leftover data to the covariance.
    if (current_.count() != 0)
        add_bundle();

    // swap data with result
    result.store_.reset();
    result.store_.swap(store_);

    // post-process data
    result.store_->convert_to_mean();
}

template <typename T, typename Str>
void cov_acc<T,Str>::add_bundle()
{
    // add batch to average and squared
    store_->data().noalias() += current_.sum();
    store_->data2().noalias() +=
                internal::outer<bind<Str, T> >(current_.sum(), current_.sum())
                / current_.count();
    store_->count() += current_.count();
    store_->count2() += current_.count() * current_.count();

    // TODO: add possibility for uplevel also here
    current_.reset();
}

template class cov_acc<double>;
template class cov_acc<std::complex<double>, circular_var>;
template class cov_acc<std::complex<double>, elliptic_var>;


// We need an explicit copy constructor, as we need to copy the data
template <typename T, typename Str>
cov_result<T,Str>::cov_result(const cov_result &other)
    : store_(other.store_ ? new cov_data<T,Str>(*other.store_) : nullptr)
{ }

template <typename T, typename Str>
cov_result<T,Str> &cov_result<T,Str>::operator=(const cov_result &other)
{
    store_.reset(other.store_ ? new cov_data<T,Str>(*other.store_) : nullptr);
    return *this;
}

template <typename T, typename Strategy>
bool operator==(const cov_result<T,Strategy> &r1, const cov_result<T,Strategy> &r2)
{
    if (r1.count() == 0 && r2.count() == 0)
        return true;

    return r1.count() == r2.count()
        && r1.count2() == r2.count2()
        && r1.store().data() == r2.store().data()
        && r1.store().data2() == r2.store().data2();
}

template bool operator==(const cov_result<double> &r1, const cov_result<double> &r2);
template bool operator==(const cov_result<std::complex<double>, circular_var> &r1,
                         const cov_result<std::complex<double>, circular_var> &r2);
template bool operator==(const cov_result<std::complex<double>, elliptic_var> &r1,
                         const cov_result<std::complex<double>, elliptic_var> &r2);

template <typename T, typename Str>
column<typename cov_result<T,Str>::var_type> cov_result<T,Str>::stderror() const
{
    internal::check_valid(*this);
    return (store_->data2().diagonal().real() / observations()).cwiseSqrt();
}

template <typename T, typename Str>
void cov_result<T,Str>::reduce(const reducer &r, bool pre_commit, bool post_commit)
{
    internal::check_valid(*this);

    if (pre_commit) {
        store_->convert_to_sum();
        r.reduce(view<T>(store_->data().data(), store_->data().rows()));
        r.reduce(view<cov_type>(store_->data2().data(), store_->data2().size()));
        r.reduce(view<size_t>(&store_->count(), 1));
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

template class cov_result<double>;
template class cov_result<std::complex<double>, circular_var>;
template class cov_result<std::complex<double>, elliptic_var>;


template <typename T, typename Str>
void serialize(serializer &s, const std::string &key, const cov_result<T,Str> &self)
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
    serialize(s, "cov", self.store_->data2_);
}

template <typename T, typename Str>
void deserialize(deserializer &s, const std::string &key, cov_result<T,Str> &self)
{
    typedef typename cov_result<T,Str>::var_type var_type;
    internal::deserializer_sentry group(s, key);

    // first deserialize the fundamentals and make sure that the target fits
    size_t new_size;
    deserialize(s, "@size", new_size);
    if (!self.valid() || self.size() != new_size)
        self.store_.reset(new cov_data<T,Str>(new_size));

    // deserialize data
    deserialize(s, "count", self.store_->count_);
    deserialize(s, "count2", self.store_->count2_);
    s.enter("mean");
    deserialize(s, "value", self.store_->data_);
    s.read("error", ndview<var_type>(nullptr, &new_size, 1)); // discard
    s.exit();
    deserialize(s, "cov", self.store_->data2_);
}

template void serialize(serializer &, const std::string &key, const cov_result<double, circular_var> &);
template void serialize(serializer &, const std::string &key, const cov_result<std::complex<double>, circular_var> &);
template void serialize(serializer &, const std::string &key, const cov_result<std::complex<double>, elliptic_var> &);

template void deserialize(deserializer &, const std::string &key, cov_result<double, circular_var> &);
template void deserialize(deserializer &, const std::string &key, cov_result<std::complex<double>, circular_var> &);
template void deserialize(deserializer &, const std::string &key, cov_result<std::complex<double>, elliptic_var> &);


template <typename T, typename Str>
std::ostream &operator<<(std::ostream &str, const cov_result<T,Str> &self)
{
    internal::format_sentry sentry(str);
    verbosity verb = internal::get_format(str, PRINT_TERSE);

    if (verb == PRINT_VERBOSE)
        str << "<X> = ";
    str << self.mean() << " +- " << self.stderror();
    if (verb == PRINT_VERBOSE)
        str << "\nSigma = " << self.cov();
    return str;
}

template std::ostream &operator<<(std::ostream &, const cov_result<double, circular_var> &);
template std::ostream &operator<<(std::ostream &, const cov_result<std::complex<double>, circular_var> &);
template std::ostream &operator<<(std::ostream &, const cov_result<std::complex<double>, elliptic_var> &);

}} /* namespace alps::alea */
