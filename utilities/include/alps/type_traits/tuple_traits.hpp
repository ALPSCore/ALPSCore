/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_TUPLE_TRAITS_HPP
#define ALPSCORE_TUPLE_TRAITS_HPP

#include <tuple>

#include <alps/type_traits/common.hpp>

namespace alps {
  template <size_t Cut, typename... T, size_t... I>
  auto tuple_tail_(const std::tuple < T... > &t, index_sequence < I... > s) -> DECLTYPE(std::make_tuple(std::get<Cut + I>(t)...));

  /// iterate over the index sequence to extract a tail of Trim size.
  /// Intel 15 is too stupid to understand size of parameter pack based on the provided tuple.
  /// So we need to provide
  template <size_t Trim, size_t Count, typename T>
  auto tuple_tail(T &t) -> DECLTYPE(tuple_tail_<Trim>(t, make_index_sequence<Count - Trim>()));

  /// iterate over the index sequence to extract a tail of Trim size
  template <size_t Trim, typename... T>
  auto tuple_tail(const std::tuple < T... > &t) -> DECLTYPE(tuple_tail_<Trim>(t, make_index_sequence<std::tuple_size<std::tuple < T... >  >::value - Trim>()));

}
#endif //ALPSCORE_TUPLE_TRAITS_HPP
