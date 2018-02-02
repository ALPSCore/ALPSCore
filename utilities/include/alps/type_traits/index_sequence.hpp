/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_INDEX_SEQUENCE_HPP
#define ALPSCORE_INDEX_SEQUENCE_HPP

#include <type_traits>

namespace alps{
  /// C++11 implementation of C++14 index sequence
  /// create argument pack of type T elements
  template<typename T, T... I>
  struct integer_sequence {
    /// type of sequence
    using value_type = T;
    /// size of sequence
    static constexpr std::size_t size() noexcept {
      return sizeof...(I);
    }
  };

  namespace integer_sequence_detail {
    template <typename, typename> struct concat;

    template <typename T, T... A, T... B>
    struct concat<integer_sequence<T, A...>, integer_sequence<T, B...>> {
      typedef integer_sequence<T, A..., B...> type;
    };

    template <typename T, size_t First, size_t Count>
    struct build_helper {
      using type = typename concat<
        typename build_helper<T, First,           Count/2>::type,
        typename build_helper<T, First + Count/2, Count - Count/2>::type
      >::type;
    };

    template <typename T, size_t First>
    struct build_helper<T, First, 1> {
      using type = integer_sequence<T, T(First)>;
    };

    template <typename T, size_t First>
    struct build_helper<T, First, 0> {
      using type = integer_sequence<T>;
    };

    template <typename T, T N>
    using builder = typename build_helper<T, 0, N>::type;
  } // namespace integer_sequence_detail

  template <typename T, T N>
  using make_integer_sequence = integer_sequence_detail::builder<T, N>;

  template <std::size_t... I>
  using index_sequence = integer_sequence<std::size_t, I...>;

  template<size_t N>
  using make_index_sequence = make_integer_sequence<size_t, N>;


}

#endif //ALPSCORE_INDEX_SEQUENCE_HPP
