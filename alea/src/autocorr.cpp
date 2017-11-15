#include <alps/alea/autocorr.hpp>

namespace alps { namespace alea {

template <typename T>
autocorr_acc<T>::autocorr_acc(size_t size, size_t batch_size, size_t granularity)
    : count_(0)
    , nextlevel_(batch_size)
    , granularity_(granularity)
    , level_()
{
    level_.push_back(var_acc<T>(size, batch_size));
}

template <typename T>
void autocorr_acc<T>::add_level()
{
    // add a new level on top and push back the
    level_.push_back(var_acc<T>(size(), granularity_));
    nextlevel_ *= granularity_;

    // make sure all the links still work (vector may reallocate elements)
    for (unsigned i = 0; i != level_.size() - 1; ++i)
        level_[i].uplevel(level_[i+1]);
}

template <typename T>
autocorr_acc<T> &autocorr_acc<T>::operator<<(const computed<T> &source)
{
    assert(count_ < nextlevel_);

    // if we require next level, then do it!
    if(++count_ == nextlevel_)
        add_level();

    // now add current element at the bottom and watch it propagate
    level_[0] << source;
    return *this;
}

template <typename T>
size_t autocorr_acc<T>::find_level(size_t min_samples) const
{
    for (unsigned i = num_level(); i != 0; --i) {
        if (level(i - 1).count() >= min_samples)
            return i - 1;
    }
    return 0;
}

template <typename T>
size_t autocorr_acc<T>::batch_size(size_t level) const
{
    size_t res = 1;
    for (unsigned i = 0; i != level + 1; ++i)
        res *= level_[i].current().capacity();
    return res;
}

template <typename T>
column<T> autocorr_acc<T>::mean() const
{
    return level_[0].result().mean();
}

template <typename T>
column<typename autocorr_acc<T>::var_type> autocorr_acc<T>::var() const
{
    return level_[find_level(256)].result().var();
}

template <typename T>
column<typename autocorr_acc<T>::var_type> autocorr_acc<T>::stderror() const
{
    return level_[find_level(256)].result().stderror();
}

template <typename T>
void autocorr_acc<T>::get_tau(sink<var_type> out) const
{
    size_t lvl = find_level(256);

    const column<var_type> &var0 = level_[0].result().var();
    const column<var_type> &varn = level_[lvl].result().var();

    // The factor `n` comes from the fact that the variance of an n-element mean
    // estimator has tighter variance by the CLT; it can be dropped if one
    // performs the batch sum rather than the batch mean.
    typename eigen<var_type>::col_map out_vec(out.data(), out.size());
    out_vec += batch_size(lvl)/batch_size(0) * varn.cwiseQuotient(var0);
}

template class autocorr_acc<double>;
template class autocorr_acc<std::complex<double> >;

}}

