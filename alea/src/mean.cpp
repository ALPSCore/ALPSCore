#include <alps/alea/mean.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/computed.hpp>

#include <iostream> // FIXME

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
    data_ *= count_;
}

template class mean_data<double>;
template class mean_data<std::complex<double> >;


// We need an explicit copy constructor, as we need to copy the data
template <typename T>
mean_acc<T>::mean_acc(const mean_acc &other)
    : store_(other.store_ ? new mean_data<T>(*other.store_) : NULL)
    , size_(other.size_)
{ }

template <typename T>
mean_acc<T> &mean_acc<T>::operator=(const mean_acc &other)
{
    store_.reset(other.store_ ? new mean_data<T>(*other.store_) : NULL);
    size_ = other.size_;
    return *this;
}

template <typename T>
mean_acc<T> &mean_acc<T>::operator<<(const computed<T> &source)
{
    if (!valid())
        throw std::runtime_error("Invalid accumulator");

    source.add_to(sink<T>(store_->data().data(), size()));
    store_->count() += 1.0;
    return *this;
}

template <typename T>
void mean_acc<T>::reset()
{
    if (size_ == size_t(-1))
        throw std::runtime_error("Uninitialized (default ctr'ed) accumulator");

    if (valid())
        store_->reset();
    else
        store_.reset(new mean_data<T>(size_));
}

template <typename T>
mean_result<T> mean_acc<T>::result() const
{
    mean_result<T> result(*store_);
    result.store_->convert_to_mean();
    return result;
}

template <typename T>
mean_result<T> mean_acc<T>::finalize()
{
    mean_result<T> result(*store_);
    result.store_.swap(store_);
    result.store_->convert_to_mean();
    return result;
}

template class mean_acc<double>;
template class mean_acc<std::complex<double> >;


template <typename T>
void mean_result<T>::reduce(reducer &r)
{
    store_->convert_to_sum();
    reducer_setup setup = r.get_setup();
    r.reduce(sink<T>(store_->data().data(), store_->data().rows()));
    r.reduce(sink<size_t>(&store_->count(), 1));
    r.commit();

    if (setup.have_result)
        store_->convert_to_mean();
    else
        store_.reset();   // free data
}

template class mean_result<double>;
template class mean_result<std::complex<double> >;


}} /* namespace alps::alea */
