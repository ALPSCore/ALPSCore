/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/mean.hpp>
#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>
#include <alps/alea/autocorr.hpp>
#include <alps/alea/batch.hpp>

#include <alps/alea/internal/util.hpp>

namespace alps { namespace alea {

// TODO: refactor this

/**
 * Stores if downgrade is valid.
 */
template <typename T, typename U>
struct can_downgrade : std::false_type { };

template <typename T>
struct can_downgrade<mean_data<T>, var_data<T> > : std::true_type { };


/**
 * Downgrades one type of data/result to another one.
 */
template <typename T, typename U>
T downgrade(const U &obj);

template <typename T, typename Str>
mean_data<T> downgrade(const var_data<T,Str> &obj)
{
    internal::check_valid(obj);

    mean_data<T> res(obj.size());
    res.data() = obj.data();
    res.count() = obj.count();
    return res;
}

template <typename T, typename Str>
mean_result<T> downgrade(const var_result<T,Str> &obj)
{
    return mean_result<T>(downgrade(obj.data()));
}




}}
