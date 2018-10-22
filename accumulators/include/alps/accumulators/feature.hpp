/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>
#include <alps/numeric/inf.hpp>
#include <alps/numeric/check_size.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/numeric/boost_array_functions.hpp>

#include "alps/numeric/scalar.hpp"
#include "alps/type_traits/is_scalar.hpp"

#include <boost/utility.hpp>

#include <type_traits>

#ifdef ALPS_HAVE_MPI
    #include <alps/hdf5/archive.hpp>
    #include <alps/accumulators/mpi.hpp>
#endif

namespace alps {
    namespace accumulators {

        template<typename T, typename F> struct has_feature
            : std::false_type
        {};

        template<typename T> struct has_result_type {
            template<typename U> static char check(typename U::result_type *);
            template<typename U> static double check(...);
            typedef std::integral_constant<bool, sizeof(char) == sizeof(check<T>(0))> type;
            constexpr static bool value = type::value;
        };

        #define NUMERIC_FUNCTION_OPERATOR(OP_NAME, OP, OP_TOKEN)                                                                               \
            namespace detail {                                                                                                                 \
                using ::alps::numeric:: OP_NAME ;                                                                                         \
                template<typename T, typename U> struct has_operator_ ## OP_TOKEN ## _impl {                                                   \
                    template<typename R> static char helper(R);                                                                                \
                    template<typename C, typename D> static char check(std::integral_constant<std::size_t, sizeof(helper(C() OP D()))>*);    \
                    template<typename C, typename D> static double check(...);                                                                 \
                    typedef std::integral_constant<bool, sizeof(char) == sizeof(check<T, U>(0))> type;                                       \
                };                                                                                                                             \
            }                                                                                                                                  \
            template<typename T, typename U> struct has_operator_ ## OP_TOKEN : public detail::has_operator_ ## OP_TOKEN ## _impl<T, U> {};

        NUMERIC_FUNCTION_OPERATOR(operator+, +, add)
        NUMERIC_FUNCTION_OPERATOR(operator-, -, sub)
        NUMERIC_FUNCTION_OPERATOR(operator*, *, mul)
        NUMERIC_FUNCTION_OPERATOR(operator/, /, div)
        #undef NUMERIC_FUNCTION_OPERATOR

        template<typename T> struct value_type {
            typedef typename T::value_type type;
        };

        namespace detail {

            /// make R<SCALAR<T>,F,B::scalar_result_type> from T if T is non-scalar, otherwise `void`
            template <template<typename,typename,typename> class R,
                      typename T, typename F, typename B>
            class make_scalar_result_type {
                typedef typename alps::numeric::scalar<T>::type scalar_type_;
                typedef typename B::scalar_result_type parent_scalar_result_type_;
                typedef R<scalar_type_, F, parent_scalar_result_type_> this_scalar_result_type_;
                public:
                typedef typename std::conditional<alps::is_scalar<T>::value,
                                                 void,
                                                 this_scalar_result_type_>::type type;
            };
        }

        namespace impl {

            template<typename T> struct ResultBase {
                typedef T value_type;
                typedef typename std::conditional<alps::is_scalar<T>::value,
                                                 void,
                                                 ResultBase<typename alps::numeric::scalar<T>::type>
                                                >::type scalar_result_type;

                /// Dummy function for merging results (always throws an exception)
                template <typename A>
                void merge(const A& /*rhs*/) {
                     throw std::runtime_error("A result cannot be merged " + ALPS_STACKTRACE);
                }

#ifdef ALPS_HAVE_MPI
                inline void collective_merge(
                      alps::mpi::communicator const & /*comm*/
                    , int /*root*/
                ) const {
                    throw std::logic_error("A result cannot be merged " + ALPS_STACKTRACE);
                }
#endif

                template<typename U> void operator+=(U const &) {}
                template<typename U> void operator-=(U const &) {}
                template<typename U> void operator*=(U const &) {}
                template<typename U> void operator/=(U const &) {}
                void negate() {}
                void inverse() {}

                void sin() {}
                void cos() {}
                void tan() {}
                void sinh() {}
                void cosh() {}
                void tanh() {}
                void asin() {}
                void acos() {}
                void atan() {}
                void abs() {}
                void sq() {}
                void sqrt() {}
                void cb() {}
                void cbrt() {}
                void exp() {}
                void log() {}
            };

            template<typename T> class AccumulatorBase {
                public:
                    typedef T value_type;
                    typedef ResultBase<T> result_type;

                    template<typename U> void operator+=(U) {
                        throw std::runtime_error("The Function operator += is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                    }
                    template<typename U> void operator-=(U) {
                        throw std::runtime_error("The Function operator -= is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                    }
                    template<typename U> void operator*=(U) {
                        throw std::runtime_error("The Function operator *= is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                    }
                    template<typename U> void operator/=(U) {
                        throw std::runtime_error("The Function operator /= is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                    }
                    void negate() {
                        throw std::runtime_error("The Function negate is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                    }
                    void inverse() {
                        throw std::runtime_error("The Function inverse is not implemented for accumulators, only for results" + ALPS_STACKTRACE);
                    }

                    void sin() { throw std::runtime_error("The Function sin is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void cos() { throw std::runtime_error("The Function cos is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void tan() { throw std::runtime_error("The Function tan is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void sinh() { throw std::runtime_error("The Function sinh is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void cosh() { throw std::runtime_error("The Function cosh is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void tanh() { throw std::runtime_error("The Function tanh is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void asin() { throw std::runtime_error("The Function asin is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void acos() { throw std::runtime_error("The Function acos is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void atan() { throw std::runtime_error("The Function atan is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void abs() { throw std::runtime_error("The Function ags is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void sq() { throw std::runtime_error("The Function sq is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void sqrt() { throw std::runtime_error("The Function sqrt is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void cb() { throw std::runtime_error("The Function cb is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void cbrt() { throw std::runtime_error("The Function cbrt is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void exp() { throw std::runtime_error("The Function exp is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }
                    void log() { throw std::runtime_error("The Function log is not implemented for accumulators, only for results" + ALPS_STACKTRACE); }

#ifdef ALPS_HAVE_MPI
                protected:
                    template <typename U, typename Op> void static reduce_if(
                          alps::mpi::communicator const & comm
                        , U const & arg
                        , U & res
                        , Op op
                        , typename std::enable_if<std::is_scalar<typename alps::hdf5::scalar_type<U>::type>::value, int>::type root
                    ) {
                        alps::alps_mpi::reduce(comm, arg, res, op, root);
                    }
                    template <typename U, typename Op> void static reduce_if(
                          alps::mpi::communicator const &
                        , U const &
                        , U &
                        , Op
                        , typename std::enable_if<!std::is_scalar<typename alps::hdf5::scalar_type<U>::type>::value, int>::type
                    ) {
                        throw std::logic_error("No alps::mpi::reduce available for this type " + std::string(typeid(U).name()) + ALPS_STACKTRACE);
                    }

                    template <typename U, typename Op> void static reduce_if(
                          alps::mpi::communicator const & comm
                        , U const & arg
                        , Op op
                        , typename std::enable_if<std::is_scalar<typename alps::hdf5::scalar_type<U>::type>::value, int>::type root
                    ) {
                        alps::alps_mpi::reduce(comm, arg, op, root);
                    }
                    template <typename U, typename Op> void static reduce_if(
                          alps::mpi::communicator const &
                        , U const &
                        , Op
                        , typename std::enable_if<!std::is_scalar<typename alps::hdf5::scalar_type<U>::type>::value, int>::type
                    ) {
                        throw std::logic_error("No alps::mpi::reduce available for this type " + std::string(typeid(U).name()) + ALPS_STACKTRACE);
                    }
#endif
            };

            template<typename T, typename F, typename B> class Accumulator {};

            template<typename T, typename F, typename B> class Result {};

            template<typename T, typename F, typename B> class BaseWrapper {};

            template<typename A, typename F, typename B> class DerivedWrapper {};

            template<typename T> struct is_accumulator : public std::false_type {};
            template<typename T, typename tag, typename B> struct is_accumulator<Accumulator<T, tag, B> > : public std::true_type {};

        }
    }
}
