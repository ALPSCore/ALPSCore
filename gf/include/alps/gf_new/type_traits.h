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
      /// check wheater type is real or complex
      template<class T> struct is_complex : std::false_type {};
      template<class T> struct is_complex<std::complex<T>> : std::true_type {};

      // following is from @href{https://gist.github.com/jappa/62f30b6da5adea60bad3}
      // we need this for two things:
      //  1. get the tail from tuple object
      //  2. create type of Green's function view
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


      template <int Cut, typename... T, std::size_t... I>
      auto subtuple_(const std::tuple<T...>& t, index_sequence<I...> s) -> decltype(std::make_tuple(std::get<Cut + I>(t)...)) {
        return std::make_tuple(std::get<Cut + I>(t)...);
      }

      // using the index sequence iterate over tuple from the end and create a new tuple
      template <int Trim, typename... T>
      auto subtuple(const std::tuple<T...>& t) -> decltype(subtuple_<Trim>(t, make_index_sequence<sizeof...(T) - Trim>()))
      {
        return subtuple_<Trim>(t, make_index_sequence<sizeof...(T) - Trim>());
      }
    }
  }
}

#endif //GREENSFUNCTIONS_TYPE_TRAITS_H
