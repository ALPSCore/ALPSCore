/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
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

    bool valid() const;

    size_t count() const;

    template <typename T>
    column<T> mean() const;

    template <typename T, typename Str=circular_var>
    column<typename Str::var_type> var() const;

    template <typename T, typename Str=circular_var>
    typename eigen<typename Str::cov_type>::matrix cov() const;

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
