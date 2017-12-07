//
// Created by iskakoff on 06/12/17.
//

#ifndef GREENSFUNCTIONS_TYPE_TRAITS_H
#define GREENSFUNCTIONS_TYPE_TRAITS_H

#include <complex>
#include <tuple>
#include <type_traits>
namespace alps {
  namespace gf {
    namespace detail {
template<class T> struct is_complex : std::false_type {};
template<class T> struct is_complex<std::complex<T>> : std::true_type {};

template<typename T, T... I>
struct integer_sequence {
  using value_type = T;

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

  template <typename T, int First, int Count>
  struct build_helper {
    using type = typename concat<
      typename build_helper<T, First,           Count/2>::type,
      typename build_helper<T, First + Count/2, Count - Count/2>::type
    >::type;
  };

  template <typename T, int First>
  struct build_helper<T, First, 1> {
    using type = integer_sequence<T, T(First)>;
  };

  template <typename T, int First>
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


template <typename... T, std::size_t... I>
auto subtuple_(const std::tuple<T...>& t, index_sequence<I...>) -> decltype(std::make_tuple(std::get<sizeof...(T) - I - 1>(t)...))
{
  return std::make_tuple(std::get<sizeof...(T) - I - 1>(t)...);
}

template <int Trim, typename... T>
auto subtuple(const std::tuple<T...>& t) -> decltype(subtuple_(t, make_index_sequence<sizeof...(T) - Trim>()))
{
  return subtuple_(t, make_index_sequence<sizeof...(T) - Trim>());
}
}
}
}

#endif //GREENSFUNCTIONS_TYPE_TRAITS_H
