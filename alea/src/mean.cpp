#include <alps/alea/mean.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/computed.hpp>
#include <alps/alea/serialize.hpp>

#include <alps/alea/internal/util.hpp>
#include <alps/alea/internal/format.hpp>

namespace alps { namespace alea {

template <typename T>
void mean_data<T>::reset()
{
    data_.fill(0);
    count_ = 0;
}

template <typename T>
void mean_data<T>::convert_to_mean()
{
    data_ /= count_;
}

template <typename T>
void mean_data<T>::convert_to_sum()
{
    // Has to be handled explicitly because of the NaNs
    if (count_ == 0) {
        reset();
        return;
    }

    data_ *= count_;
}

template class mean_data<double>;
template class mean_data<std::complex<double> >;


// We need an explicit copy constructor, as we need to copy the data
template <typename T>
mean_acc<T>::mean_acc(const mean_acc &other)
    : store_(other.store_ ? new mean_data<T>(*other.store_) : nullptr)
    , size_(other.size_)
{ }

template <typename T>
mean_acc<T> &mean_acc<T>::operator=(const mean_acc &other)
{
    store_.reset(other.store_ ? new mean_data<T>(*other.store_) : nullptr);
    size_ = other.size_;
    return *this;
}

template <typename T>
void mean_acc<T>::add(const computed<T> &source, size_t count)
{
    internal::check_valid(*this);
    source.add_to(view<T>(store_->data().data(), size()));
    store_->count() += count;
}

template <typename T>
mean_acc<T> &mean_acc<T>::operator<<(const mean_result<T> &other)
{
    internal::check_valid(*this);
    if (size() != other.size())
        throw size_mismatch();

    // HACK we need this for "outwardly constant" manipulation
    mean_data<T> &other_store = const_cast<mean_data<T> &>(other.store());
    other_store.convert_to_sum();
    store_->data() += other_store.data();
    store_->count() += other_store.count();
    other_store.convert_to_mean();
    return *this;
}

template <typename T>
void mean_acc<T>::reset()
{
    if (valid())
        store_->reset();
    else
        store_.reset(new mean_data<T>(size_));
}

template <typename T>
mean_result<T> mean_acc<T>::result() const
{
    internal::check_valid(*this);
    mean_result<T> result(*store_);
    result.store_->convert_to_mean();
    return result;
}

template <typename T>
mean_result<T> mean_acc<T>::finalize()
{
    mean_result<T> result;
    finalize_to(result);
    return result;
}

template <typename T>
void mean_acc<T>::finalize_to(mean_result<T> &result)
{
    internal::check_valid(*this);
    result.store_.reset();
    result.store_.swap(store_);
    result.store_->convert_to_mean();
}

template class mean_acc<double>;
template class mean_acc<std::complex<double> >;


// We need an explicit copy constructor, as we need to copy the data
template <typename T>
mean_result<T>::mean_result(const mean_result &other)
    : store_(other.store_ ? new mean_data<T>(*other.store_) : nullptr)
{ }

template <typename T>
mean_result<T> &mean_result<T>::operator=(const mean_result &other)
{
    store_.reset(other.store_ ? new mean_data<T>(*other.store_) : nullptr);
    return *this;
}

template <typename T>
bool operator==(const mean_result<T> &r1, const mean_result<T> &r2)
{
    if (r1.count() == 0 && r2.count() == 0)
        return true;

    return r1.count() == r2.count()
        && r1.store().data() == r2.store().data();
}

template bool operator==(const mean_result<double> &r1,
                         const mean_result<double> &r2);
template bool operator==(const mean_result<std::complex<double>> &r1,
                         const mean_result<std::complex<double>> &r2);

template <typename T>
void mean_result<T>::reduce(const reducer &r, bool pre_commit, bool post_commit)
{
    internal::check_valid(*this);
    if (pre_commit) {
        store_->convert_to_sum();
        r.reduce(view<T>(store_->data().data(), store_->data().rows()));
        r.reduce(view<size_t>(&store_->count(), 1));
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

template class mean_result<double>;
template class mean_result<std::complex<double> >;


template <typename T>
void serialize(serializer &s, const std::string &key, const mean_result<T> &self)
{
    internal::check_valid(self);
    internal::serializer_sentry group(s, key);

    serialize(s, "@size", self.store_->data_.size());
    serialize(s, "count", self.store_->count_);
    s.enter("mean");
    serialize(s, "value", self.store_->data_);
    s.exit();
}

template <typename T>
void deserialize(deserializer &s, const std::string &key, mean_result<T> &self)
{
    internal::deserializer_sentry group(s, key);

    // first deserialize the fundamentals and make sure that the target fits
    size_t new_size;
    deserialize(s, "@size", new_size);
    if (!self.valid() || self.size() != new_size)
        self.store_.reset(new mean_data<T>(new_size));

    // deserialize data
    deserialize(s, "count", self.store_->count_);
    s.enter("mean");
    deserialize(s, "value", self.store_->data_);
    s.exit();
}

template void serialize(serializer &, const std::string &, const mean_result<double> &);
template void serialize(serializer &, const std::string &, const mean_result<std::complex<double> > &);

template void deserialize(deserializer &, const std::string &, mean_result<double> &);
template void deserialize(deserializer &, const std::string &, mean_result<std::complex<double> > &);


template <typename T>
std::ostream &operator<<(std::ostream &str, const mean_result<T> &self)
{
    internal::check_valid(self);
    internal::format_sentry sentry(str);
    verbosity verb = internal::get_format(str, PRINT_TERSE);

    if (verb == PRINT_VERBOSE)
        str << "<X> = ";
    str << self.mean();
    return str;
}

template std::ostream &operator<<(std::ostream &, const mean_result<double> &);
template std::ostream &operator<<(std::ostream &, const mean_result<std::complex<double>> &);


}} /* namespace alps::alea */
