#include <alps/alea/mean.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/computed.hpp>

namespace alps { namespace alea {

template <typename T>
void mean_data<T>::reset()
{
    data_.fill(0);
    count_ = 0;
    state_ = SUM;
}

template <typename T>
void mean_data<T>::state(data_state new_state)
{
    if (new_state == state_)
        return;
    if (new_state == SUM)
        data_ *= count_;
    else // if (state_ == SUM)
        data_ /= count_;
    state_ = new_state;
}

template <typename T>
void mean_data<T>::unlock_mean() const
{
    if (state_ == SUM) {
        data_ /= count_;
        state_ = MEAN;
    }
}

template <typename T>
void mean_data<T>::unlock_sum() const
{
    if (state_ == MEAN) {
        data_ *= count_;
        state_ = SUM;
    }
}

template <typename T>
void mean_data<T>::reduce(reducer &r)
{
    unlock_sum();
    reducer::setup setup = r.begin();
    r.reduce(sink<T>(data_.data(), data_.rows()));
    r.reduce(sink<size_t>(&count_, 1));
    r.commit();

    if (setup.have_result)
        unlock_mean();
}

template <typename T>
void mean_data<T>::serialize(serializer &s) const
{
    // FIXME: constness
    computed_adapter<T, column<T> > data_ad(data_);
    computed_adapter<long, long> count_ad(count_);

    unlock_mean();
    s.write("mean/data", data_ad);
    s.write("count", count_ad);
}

template class mean_data<double>;
template class mean_data<std::complex<double> >;


template <typename T>
mean_acc<T> &mean_acc<T>::operator<<(computed<T> &source)
{
    store_.unlock_sum();
    source.add_to(sink<T>(store_.data().data(), store_.size()));
    store_.count() += 1.0;
    return *this;
}

template class mean_acc<double>;
template class mean_acc<std::complex<double> >;

}} /* namespace alps::alea */
