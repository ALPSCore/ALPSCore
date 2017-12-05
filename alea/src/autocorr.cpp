#include <alps/alea/autocorr.hpp>

#include <alps/alea/internal/util.hpp>

namespace alps { namespace alea {

template <typename T>
autocorr_acc<T>::autocorr_acc(size_t size, size_t batch_size, size_t granularity)
    : size_(size)
    , batch_size_(batch_size)
    , count_(0)
    , nextlevel_(batch_size)
    , granularity_(granularity)
    , level_()
{
    level_.push_back(var_acc<T>(size, batch_size));
}

template <typename T>
void autocorr_acc<T>::reset()
{
    count_ = 0;
    nextlevel_ = batch_size_;
    level_.clear();
    level_.push_back(var_acc<T>(size_, batch_size_));
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
    internal::check_valid(*this);

    // if we require next level, then do it!
    if(++count_ == nextlevel_)
        add_level();

    // now add current element at the bottom and watch it propagate
    level_[0] << source;
    return *this;
}

template <typename T>
autocorr_result<T> autocorr_acc<T>::result() const
{
    internal::check_valid(*this);

    autocorr_result<T> result;
    for (size_t i = 0; i != level_.size(); ++i)
        result.level_.push_back(level_[i].result());
    return result;
}

template <typename T>
autocorr_result<T> autocorr_acc<T>::finalize()
{
    autocorr_result<T> result;
    finalize_to(result);
    return result;
}

template <typename T>
void autocorr_acc<T>::finalize_to(autocorr_result<T> &result)
{
    internal::check_valid(*this);
    result.level_.resize(level_.size());
    for (size_t i = 0; i != level_.size(); ++i)
        level_[i].finalize_to(result.level_[i]);

    level_.clear();     // signal invalidity
}

template class autocorr_acc<double>;
template class autocorr_acc<std::complex<double> >;


template <typename T>
size_t autocorr_result<T>::batch_size(size_t i) const
{
    // This is a little hairy, but it works in practice
    return level_[0].count() / level_[i].count();
}

template <typename T>
size_t autocorr_result<T>::find_level(size_t min_samples) const
{
    // TODO: this can be done in O(1)
    for (unsigned i = nlevel(); i != 0; --i) {
        if (level(i - 1).count() >= min_samples)
            return i - 1;
    }
    return 0;
}

template <typename T>
column<typename autocorr_result<T>::var_type> autocorr_result<T>::var() const
{
    size_t lvl = find_level(default_min_samples);

    // The factor comes from the fact that we accumulate means of batch_size
    // elements, and therefore we get this by the law of large numbers
    return batch_size(lvl) * level_[lvl].var();
}

template <typename T>
column<typename autocorr_result<T>::var_type> autocorr_result<T>::stderror() const
{
    size_t lvl = find_level(default_min_samples);

    // Standard error of the mean has another 1/N (but at the level!)
    double fact = 1. * batch_size(lvl) / level_[lvl].count();
    return (fact * level_[lvl].var()).cwiseSqrt();
}

template <typename T>
column<typename autocorr_result<T>::var_type> autocorr_result<T>::tau() const
{
    size_t lvl = find_level(default_min_samples);
    const column<var_type> &var0 = level_[0].var();
    const column<var_type> &varn = level_[lvl].var();

    // The factor `n` comes from the fact that the variance of an n-element mean
    // estimator has tighter variance by the CLT; it can be dropped if one
    // performs the batch sum rather than the batch mean.
    double fact = batch_size(lvl) / 2 * batch_size(0);
    return (fact * varn.array()/ var0.array() - 0.5).matrix();
}

template class autocorr_result<double>;
template class autocorr_result<std::complex<double> >;

}}

