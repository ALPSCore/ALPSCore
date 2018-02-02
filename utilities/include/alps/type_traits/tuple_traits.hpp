/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_TUPLE_TRAITS_HPP
#define ALPSCORE_TUPLE_TRAITS_HPP

#include <tuple>

#include <alps/type_traits/common.hpp>

namespace alps {
  template <int Cut, typename... T, std::size_t... I>
  auto tuple_tail_(const std::tuple < T... > &t, index_sequence < I... > s) -> DECLTYPE(std::make_tuple(std::get<Cut + I>(t)...));

  /// iterate over the index sequence to extract a tail of Trim size
  template <int Trim, typename... T>
  auto tuple_tail(const std::tuple < T... > &t) -> DECLTYPE(tuple_tail_<Trim>(t, make_index_sequence<sizeof...(T) - Trim>()));

}
#endif //ALPSCORE_TUPLE_TRAITS_HPP
