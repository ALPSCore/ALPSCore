/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <boost/variant.hpp>

#include <alps/alea/mean.hpp>
#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>
#include <alps/alea/autocorr.hpp>
#include <alps/alea/batch.hpp>

namespace alps { namespace alea {

class result;
void serialize(serializer &, const std::string &, const result &);

}}


namespace alps { namespace alea {

struct estimate_unavailable : public std::exception { };

struct estimate_type_mismatch : public std::exception { };

class result
{
public:
    result() : res_(mean_result<double>()) { }    // empty

    template <typename T>
    result(const mean_result<T> &res) : res_(res) { }

    template <typename T, typename Str>
    result(const var_result<T,Str> &res) : res_(res) { }

    template <typename T, typename Str>
    result(const cov_result<T,Str> &res) : res_(res) { }

    template <typename T>
    result(const autocorr_result<T> &res) : res_(res) { }

    template <typename T>
    result(const batch_result<T> &res) : res_(res) { }

    /** Returns `false` if `finalize()` has been called, `true` otherwise */
    bool valid() const;

    /** Number of components of the random vector (e.g., size of mean) */
    size_t size() const;

    /** Returns number of accumulated data points */
    size_t count() const;

    /** Returns sample mean */
    template <typename T>
    column<T> mean() const;

    /** Returns bias-corrected sample variance for given strategy */
    template <typename T, typename Str=circular_var>
    column<typename bind<Str,T>::var_type> var() const;

    /** Returns bias-corrected sample covariance matrix for given strategy */
    template <typename T, typename Str=circular_var>
    typename eigen<typename bind<Str,T>::cov_type>::matrix cov() const;

    /** Convert result to a permanent format (write to disk etc.) */
    friend void serialize(serializer &, const std::string &, const result &);

private:
    typedef boost::variant<
          mean_result<double>
        , mean_result<std::complex<double> >
        , var_result<double>
        , var_result<std::complex<double> >
        , var_result<std::complex<double>, elliptic_var>
        , cov_result<double>
        , cov_result<std::complex<double> >
        , cov_result<std::complex<double>, elliptic_var>
        , autocorr_result<double>
        , autocorr_result<std::complex<double> >
        , batch_result<double>
        , batch_result<std::complex<double> >
        > variant_type;

    variant_type res_;
};


}}
