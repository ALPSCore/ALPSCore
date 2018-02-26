#include <alps/alea/autocorr.hpp>
#include <alps/alea/serialize.hpp>

#include <alps/alea/internal/util.hpp>
#include <alps/alea/internal/format.hpp>

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
    // add a new level on top and push back the nextlevel
    nextlevel_ *= granularity_;
    level_.push_back(var_acc<T>(size(), nextlevel_));
}

template <typename T>
void autocorr_acc<T>::add(const computed<T> &source, size_t count)
{
    assert(count_ < nextlevel_);
    internal::check_valid(*this);

    // if we require next level, then do it!
    count_ += count;
    if(count_ >= nextlevel_)
        add_level();

    // now add current element at the bottom and watch it propagate
    level_[0].add(source, count, level_.data() + 1);
}

template <typename T>
autocorr_acc<T> &autocorr_acc<T>::operator<<(const autocorr_result<T> &other)
{
    internal::check_valid(*this);

    // ensure we have enough levels to hold other data
    for (size_t i = nlevel(); i < other.nlevel(); ++i)
        level_.push_back(var_acc<T>(size_, batch_size_));

    // merge the levels
    // FIXME handle the highers other level by doing proper mergin
    for (size_t i = 0; i != other.nlevel(); ++i)
        level_[i] << other.level(i);

    return *this;
}

template <typename T>
autocorr_result<T> autocorr_acc<T>::result() const
{
    internal::check_valid(*this);
    autocorr_result<T> result;
    autocorr_acc<T>(*this).finalize_to(result);
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

    // Finalize each level.
    // NOTE: it is imperative to do this in bottom-up order, since it collects
    //       the left-over data in the current batch from the low-lying levels
    //       and propagates them upwards.
    for (size_t i = 0; i != level_.size() - 1; ++i)
        level_[i].finalize_to(result.level_[i], level_.data() + i + 1);

    level_[nlevel() - 1].finalize_to(result.level_[nlevel() - 1], nullptr);
    level_.clear();     // signal invalidity
}

template class autocorr_acc<double>;
template class autocorr_acc<std::complex<double> >;

template <typename T>
bool operator==(const autocorr_result<T> &r1, const autocorr_result<T> &r2)
{
    if(r1.nlevel() != r2.nlevel()) return false;
    for(size_t i = 0; i < r1.nlevel(); ++i) {
        if(r1.level(i) != r2.level(i))
            return false;
    }
    return true;
}

template bool operator==(const autocorr_result<double> &r1,
                         const autocorr_result<double> &r2);
template bool operator==(const autocorr_result<std::complex<double>> &r1,
                         const autocorr_result<std::complex<double>> &r2);

template <typename T>
size_t autocorr_result<T>::batch_size(size_t i) const
{
    return level_[i].batch_size();
}

template <typename T>
size_t autocorr_result<T>::find_level(size_t min_samples) const
{
    // TODO: this can be done in O(1)
    for (unsigned i = nlevel(); i != 0; --i) {
        if (level(i-1).count() / level(i-1).batch_size() >= min_samples)
            return i - 1;
    }
    return 0;
}

template <typename T>
column<typename autocorr_result<T>::var_type> autocorr_result<T>::var() const
{
    size_t lvl = find_level(DEFAULT_MIN_SAMPLES);

    // The factor comes from the fact that we accumulate sums of batch_size
    // elements, and therefore we get this by the law of large numbers
    return level_[lvl].var();
}

template <typename T>
column<typename autocorr_result<T>::var_type> autocorr_result<T>::stderror() const
{
    size_t lvl = find_level(DEFAULT_MIN_SAMPLES);

    // Standard error of the mean has another 1/N (but at the level!)
    //double fact = 1. * batch_size(lvl) / level_[lvl].count();
    //return (fact * level_[lvl].var()).cwiseSqrt();
    return level_[lvl].stderror();
}

template <typename T>
column<typename autocorr_result<T>::var_type> autocorr_result<T>::tau() const
{
    size_t lvl = find_level(DEFAULT_MIN_SAMPLES);
    const column<var_type> &var0 = level_[0].var();
    const column<var_type> &varn = level_[lvl].var();

    return (0.5 * varn.array() / var0.array() - 0.5).matrix();
}

template <typename T>
void autocorr_result<T>::reduce(const reducer &r, bool pre_commit, bool post_commit)
{
    internal::check_valid(*this);

    if (pre_commit) {
        // initialize reduction: we may need to amend the number of levels
        size_t needs_levels = r.get_max(nlevel());
        for (size_t i = nlevel(); i != needs_levels; ++i)
            level_.push_back(level_result_type(var_data<T>(size())));

        // TODO: figure out if this is statistically sound
        for (size_t i = 0; i != nlevel(); ++i)
            level_[i].reduce(r, true, false);
    }
    if (pre_commit && post_commit) {
        // perform commit
        r.commit();
    }
    if (post_commit) {
        // cleanups
        reducer_setup setup = r.get_setup();
        for (size_t i = 0; i != nlevel(); ++i)
            level_[i].reduce(r, false, true);
        if (!setup.have_result)
            level_.clear();         // invalidate
    }
}

template class autocorr_result<double>;
template class autocorr_result<std::complex<double> >;


template <typename T>
void serialize(serializer &s, const std::string &key, const autocorr_result<T> &self)
{
    internal::check_valid(self);
    internal::serializer_sentry group(s, key);
    serialize(s, "@size", self.size());
    serialize(s, "@nlevel", self.nlevel());

    s.enter("level");
    for (size_t i = 0; i != self.nlevel(); ++i)
        serialize(s, std::to_string(i), self.level_[i]);
    s.exit();

    s.enter("mean");
    serialize(s, "value", self.mean());
    serialize(s, "error", self.stderror());
    s.exit();
}

template <typename T>
void deserialize(deserializer &s, const std::string &key, autocorr_result<T> &self)
{
    typedef typename autocorr_result<T>::var_type var_type;
    internal::deserializer_sentry group(s, key);

    // first deserialize the fundamentals and make sure that the target fits
    size_t new_size = 1;
    s.read("@size", ndview<size_t>(nullptr, &new_size, 0)); // discard
    size_t new_nlevel;
    deserialize(s, "@nlevel", new_nlevel);
    self.level_.resize(new_nlevel);

    s.enter("level");
    for (size_t i = 0; i != self.nlevel(); ++i)
        deserialize(s, std::to_string(i), self.level_[i]);
    s.exit();

    s.enter("mean");
    new_size = self.size();
    s.read("value", ndview<T>(nullptr, &new_size, 1)); // discard
    s.read("error", ndview<var_type>(nullptr, &new_size, 1)); // discard
    s.exit();
}

template void serialize(serializer &, const std::string &key, const autocorr_result<double> &);
template void serialize(serializer &, const std::string &key, const autocorr_result<std::complex<double>> &);

template void deserialize(deserializer &, const std::string &key, autocorr_result<double> &);
template void deserialize(deserializer &, const std::string &key, autocorr_result<std::complex<double> > &);

template <typename T>
std::ostream &operator<<(std::ostream &str, const autocorr_result<T> &self)
{
    internal::check_valid(self);
    internal::format_sentry sentry(str);
    verbosity verb = internal::get_format(str, PRINT_TERSE);

    if (verb == PRINT_VERBOSE)
        str << "<X> = ";
    str << self.mean() << " +- " << self.stderror();

    if (verb == PRINT_VERBOSE) {
        str << "\nLevels:" << PRINT_TERSE;
        for (const var_result<T> &curr : self.level_)
            str << "\n  " << curr;
    }
    return str;
}

template std::ostream &operator<<(std::ostream &, const autocorr_result<double> &);
template std::ostream &operator<<(std::ostream &, const autocorr_result<std::complex<double>> &);

}}

