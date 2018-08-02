/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/alea/batch.hpp>
#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>
#include <alps/alea/serialize.hpp>

#include <alps/alea/internal/util.hpp>
#include <alps/alea/internal/format.hpp>

#include <numeric>

namespace alps { namespace alea {

template <typename T>
batch_data<T>::batch_data(size_t size, size_t num_batches)
    : batch_(size, num_batches)
    , count_(num_batches)
{
    reset();
}

template <typename T>
void batch_data<T>::reset()
{
    batch_.fill(0);
    count_.fill(0);
}

template class batch_data<double>;
template class batch_data<std::complex<double> >;


template <typename T>
batch_acc<T>::batch_acc(size_t size, size_t num_batches, size_t base_size)
    : size_(size)
    , num_batches_(num_batches)
    , base_size_(base_size)
    , store_(new batch_data<T>(size, num_batches))
    , cursor_(num_batches)
    , offset_(num_batches)
{
    if (num_batches % 2 != 0) {
        throw std::runtime_error("Number of batches must be even to allow "
                                 "for rebatching.");
    }
    for (size_t i = 0; i != num_batches; ++i)
        offset_[i] = i * base_size_;
}

template <typename T>
batch_acc<T>::batch_acc(const batch_acc &other)
    : size_(other.size_)
    , num_batches_(other.num_batches_)
    , base_size_(other.base_size_)
    , store_(other.store_ ? new batch_data<T>(*other.store_) : nullptr)
    , cursor_(other.cursor_)
    , offset_(other.offset_)
{ }

template <typename T>
batch_acc<T> &batch_acc<T>::operator=(const batch_acc &other)
{
    size_ = other.size_;
    num_batches_ = other.num_batches_;
    base_size_ = other.base_size_;
    store_.reset(other.store_ ? new batch_data<T>(*other.store_) : nullptr);
    cursor_ = other.cursor_;
    offset_ = other.offset_;
    return *this;
}

template <typename T>
void batch_acc<T>::reset()
{
    cursor_.reset();
    for (size_t i = 0; i != num_batches_; ++i)
        offset_[i] = i * base_size_;

    if (valid())
        store_->reset();
    else
        store_.reset(new batch_data<T>(size_, num_batches_));
}

template <typename T>
void batch_acc<T>::set_size(size_t size)
{
    size_ = size;
    if (valid()) {
        store_.reset(new batch_data<T>(size_, num_batches_));
        reset();
    }
}

template <typename T>
void batch_acc<T>::set_batch_size(size_t batch_size)
{
    base_size_ = batch_size;
    if (valid())
        reset();
}

template <typename T>
void batch_acc<T>::set_num_batches(size_t num_batches)
{
    // TODO: handle the case where we just discard levels more gracefully
    num_batches_ = num_batches;
    if (valid()) {
        store_.reset(new batch_data<T>(size_, num_batches_));
        reset();
    }
}

template <typename T>
void batch_acc<T>::add(const computed<T> &source, size_t count)
{
    internal::check_valid(*this);

    // batch is full, move the cursor.
    // Doing this before the addition ensures no empty batches.
    if (store_->count()(cursor_.current()) >= current_batch_size())
        next_batch();

    // Since Eigen matrix are column-major, we can just pass the pointer
    source.add_to(view<T>(store_->batch().col(cursor_.current()).data(), size()));
    store_->count()(cursor_.current()) += count;
}

template <typename T>
batch_acc<T> &batch_acc<T>::operator<<(const batch_result<T> &other)
{
    internal::check_valid(*this);
    if (size() != other.size())
        throw size_mismatch();
    // TODO this is not strictly speaking necessary when done properly
    if (num_batches() != other.num_batches())
        throw size_mismatch();

    // FIXME: this is a terrible idea because it mixes two time series
    batch_data<T> &other_store = const_cast<batch_data<T> &>(other.store());
    store_->batch() += other_store.batch();
    store_->count() += other_store.count();
    return *this;
}

template <typename T>
void batch_acc<T>::next_batch()
{
    ++cursor_;
    if (cursor_.merge_mode()) {
        // merge counts
        store_->count()(cursor_.merge_into()) += store_->count()(cursor_.current());
        store_->count()(cursor_.current()) = 0;

        // merge batches
        store_->batch().col(cursor_.merge_into()) +=
                                        store_->batch().col(cursor_.current());
        store_->batch().col(cursor_.current()).fill(0);

        // merge offsets
        offset_(cursor_.merge_into()) = std::min(offset_(cursor_.merge_into()),
                                                 offset_(cursor_.current()));
        offset_(cursor_.current()) = count();
    }
}

template <typename T>
batch_result<T> batch_acc<T>::result() const
{
    internal::check_valid(*this);
    batch_result<T> result(*store_);
    return result;
}

template <typename T>
batch_result<T> batch_acc<T>::finalize()
{
    batch_result<T> result;
    finalize_to(result);
    return result;
}

template <typename T>
void batch_acc<T>::finalize_to(batch_result<T> &result)
{
    internal::check_valid(*this);
    result.store_.reset();
    result.store_.swap(store_);
}

template class batch_acc<double>;
template class batch_acc<std::complex<double> >;


template <typename T>
batch_result<T>::batch_result(const batch_result &other)
    : store_(other.store_ ? new batch_data<T>(*other.store_) : nullptr)
{ }

template <typename T>
batch_result<T> &batch_result<T>::operator=(const batch_result &other)
{
    store_.reset(other.store_ ? new batch_data<T>(*other.store_) : nullptr);
    return *this;
}

template <typename T>
bool operator==(const batch_result<T> &r1, const batch_result<T> &r2)
{
    return r1.count() == r2.count()
        && r1.store().batch() == r2.store().batch();
}

template bool operator==(const batch_result<double> &r1,
                         const batch_result<double> &r2);
template bool operator==(const batch_result<std::complex<double>> &r1,
                         const batch_result<std::complex<double>> &r2);

template <typename T>
column<T> batch_result<T>::mean() const
{
    return store_->batch().rowwise().sum() / count();
}

template <typename T>
template <typename Str>
column<typename bind<Str,T>::var_type> batch_result<T>::var() const
{
    var_acc<T, Str> aux_acc(store_->size());
    for (size_t i = 0; i != store_->num_batches(); ++i) {
        aux_acc.add(make_adapter(store_->batch().col(i)), store_->count()(i),
                    nullptr);
    }
    return aux_acc.finalize().var();
}

template <typename T>
template <typename Str>
typename eigen<typename bind<Str,T>::cov_type>::matrix batch_result<T>::cov() const
{
    cov_acc<T, Str> aux_acc(store_->size());
    for (size_t i = 0; i != store_->num_batches(); ++i)
        aux_acc.add(make_adapter(store_->batch().col(i)), store_->count()(i));
    return aux_acc.finalize().cov();
}

template <typename T>
column<typename bind<circular_var,T>::var_type> batch_result<T>::stderror() const
{
    var_acc<T, circular_var> aux_acc(store_->size());
    for (size_t i = 0; i != store_->num_batches(); ++i) {
        aux_acc.add(make_adapter(store_->batch().col(i)), store_->count()(i),
                    nullptr);
    }
    return aux_acc.finalize().stderror();
}

template <typename T>
void batch_result<T>::reduce(const reducer &r, bool pre_commit, bool post_commit)
{
    // FIXME this is bad since it mixes bins
    internal::check_valid(*this);
    if (pre_commit) {
        r.reduce(view<T>(store_->batch().data(), store_->batch().size()));
        r.reduce(view<size_t>(store_->count().data(), store_->num_batches()));
    }
    if (pre_commit && post_commit) {
        r.commit();
    }
    if (post_commit) {
        reducer_setup setup = r.get_setup();
        if (!setup.have_result)
            store_.reset();   // free data
    }
}

template column<double> batch_result<double>::var<circular_var>() const;
template column<double> batch_result<std::complex<double> >::var<circular_var>() const;
template column<complex_op<double> > batch_result<std::complex<double> >::var<elliptic_var>() const;

template eigen<double>::matrix batch_result<double>::cov< circular_var>() const;
template eigen<std::complex<double>>::matrix batch_result<std::complex<double> >::cov<circular_var>() const;
template eigen<complex_op<double> >::matrix batch_result<std::complex<double> >::cov<elliptic_var>() const;

template class batch_result<double>;
template class batch_result<std::complex<double> >;


template <typename T>
void serialize(serializer &s, const std::string &key, const batch_result<T> &self)
{
    internal::check_valid(self);
    internal::serializer_sentry group(s, key);

    serialize(s, "@size", self.size());
    serialize(s, "@num_batches", self.store().num_batches());

    s.enter("batch");
    serialize(s, "count", self.store().count());
    serialize(s, "sum", self.store().batch());
    s.exit();

    s.enter("mean");
    serialize(s, "value", self.mean());
    serialize(s, "error", self.stderror());
    s.exit();
}

template <typename T>
void deserialize(deserializer &s, const std::string &key, batch_result<T> &self)
{
    typedef typename bind<circular_var, T>::var_type var_type;
    internal::deserializer_sentry group(s, key);

    // first deserialize the fundamentals and make sure that the target fits
    size_t new_size, new_nbatches;
    deserialize(s, "@size", new_size);
    deserialize(s, "@num_batches", new_nbatches);
    if (!self.valid() || self.size() != new_size || self.store().num_batches() != new_nbatches)
        self.store_.reset(new batch_data<T>(new_size, new_nbatches));

    // deserialize data
    s.enter("batch");
    deserialize(s, "count", self.store().count());
    deserialize(s, "sum", self.store().batch());
    s.exit();

    s.enter("mean");
    s.read("value", ndview<T>(nullptr, &new_size, 1)); // discard
    s.read("error", ndview<var_type>(nullptr, &new_size, 1)); // discard
    s.exit();
}

template void serialize(serializer &, const std::string &key, const batch_result<double> &);
template void serialize(serializer &, const std::string &key, const batch_result<std::complex<double>> &);

template void deserialize(deserializer &, const std::string &key, batch_result<double> &);
template void deserialize(deserializer &, const std::string &key, batch_result<std::complex<double> > &);

template <typename T>
std::ostream &operator<<(std::ostream &str, const batch_result<T> &self)
{
    internal::check_valid(self);
    internal::format_sentry sentry(str);
    verbosity verb = internal::get_format(str, PRINT_TERSE);

    if (verb == PRINT_VERBOSE)
        str << "<X> = ";
    str << self.mean() << " +- " << self.stderror();

    if (verb == PRINT_VERBOSE) {
        str << "\n<Xi> = " << self.store().batch()
            << "\nNi = " << self.store().count();
    }
    return str;
}

template std::ostream &operator<<(std::ostream &, const batch_result<double> &);
template std::ostream &operator<<(std::ostream &, const batch_result<std::complex<double>> &);

}} /* namespace alps::alea */
