#include <alps/alea/result.hpp>

#include <type_traits>

#include <iostream> // FIXME

namespace alps { namespace alea {

struct valid_visitor
{
    typedef bool result_type;           // required by boost::apply_visitor

    template <typename Res>
    bool operator() (const Res &r) const { return r.valid(); }
};

struct count_visitor
{
    typedef size_t result_type;

    template <typename Res>
    size_t operator() (const Res &r) const { return r.count(); }
};

template <typename T>          // T = double or std::complex<double>
struct mean_visitor
{
    typedef column<T> result_type;

    column<T> operator() (const mean_result<T> &r) const { return r.mean(); }
    template <typename Str>
    column<T> operator() (const var_result<T,Str> &r) const { return r.mean(); }
    template <typename Str>
    column<T> operator() (const cov_result<T,Str> &r) const { return r.mean(); }
    column<T> operator() (const autocorr_result<T> &r) const { return r.mean(); }
    column<T> operator() (const batch_result<T> &r) const { return r.mean(); }

    // default case
    template <typename Res>
    column<T> operator() (const Res &) const { throw estimate_type_mismatch(); }
};

template <typename T, typename Str>          // T = double or std::complex<double>
struct var_visitor
{
    typedef column<typename bind<Str,T>::var_type> result_type;

    result_type operator() (const mean_result<T> &) const { throw estimate_unavailable(); }
    result_type operator() (const var_result<T,Str> &r) const { return r.var(); }
    result_type operator() (const cov_result<T,Str> &r) const { return r.var(); }
    result_type operator() (const autocorr_result<T> &) const { throw estimate_type_mismatch(); }
    result_type operator() (const batch_result<T> &r) const { return r.template var<Str>(); }

    // default case
    template <typename Res>
    result_type operator() (const Res &) const { throw estimate_type_mismatch(); }
};

template <typename T>          // T = double or std::complex<double>
struct var_visitor<T, circular_var>
{
    typedef circular_var Str;
    typedef column<typename bind<circular_var,T>::var_type> result_type;

    result_type operator() (const mean_result<T> &) const { throw estimate_unavailable(); }
    result_type operator() (const var_result<T,Str> &r) const { return r.var(); }
    result_type operator() (const cov_result<T,Str> &r) const { return r.var(); }
    result_type operator() (const autocorr_result<T> &r) const { return r.var(); }
    result_type operator() (const batch_result<T> &r) const { return r.var(); }

    // default case
    template <typename Res>
    result_type operator() (const Res &) const { throw estimate_type_mismatch(); }
};

template <typename T, typename Str>          // T = double or std::complex<double>
struct cov_visitor
{
    typedef typename eigen<typename bind<Str,T>::cov_type>::matrix result_type;

    result_type operator() (const mean_result<T> &) const { throw estimate_unavailable(); }
    result_type operator() (const var_result<T,Str> &) const { throw estimate_unavailable(); }
    result_type operator() (const cov_result<T,Str> &r) const { return r.cov(); }
    result_type operator() (const autocorr_result<T> &) const { throw estimate_unavailable(); }
    result_type operator() (const batch_result<T> &r) const { r.template cov<Str>(); }

    // default case
    template <typename Res>
    result_type operator() (const Res &) const { throw estimate_type_mismatch(); }
};

bool result::valid() const
{
    return boost::apply_visitor(valid_visitor(), res_);
}

size_t result::count() const
{
    return boost::apply_visitor(count_visitor(), res_);
}

template <typename T>
column<T> result::mean() const
{
    return boost::apply_visitor(mean_visitor<T>(), res_);
}

template <typename T, typename Str>
column<typename Str::var_type> result::var() const
{
    return boost::apply_visitor(var_visitor<T,Str>(), res_);
}

template <typename T, typename Str>
typename eigen<typename Str::cov_type>::matrix result::cov() const
{
    return boost::apply_visitor(cov_visitor<T,Str>(), res_);
}

}}
