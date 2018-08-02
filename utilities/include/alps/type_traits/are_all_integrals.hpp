/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
//
// Created by Sergei on 2/22/18.
//

#ifndef ALPSCORE_ARE_ALL_INTEGRALS_H
#define ALPSCORE_ARE_ALL_INTEGRALS_H

#include <type_traits>

template<typename I0, typename... I> struct are_all_integrals :
    std::integral_constant<bool, std::is_integral<I0>::value && are_all_integrals<I...>::value>
{};
template<typename I0> struct are_all_integrals<I0> : std::is_integral<I0> {};

#endif //ALPSCORE_ARE_ALL_INTEGERS_H
