/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>

#include <alps/accumulators/feature.hpp>
#include <alps/accumulators/parameter.hpp>
#include <alps/accumulators/feature/count.hpp>
#include <alps/accumulators/archive_traits.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/numeric/inf.hpp>
#include <alps/numeric/boost_array_functions.hpp>
#include <alps/numeric/check_size.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/short_print.hpp>

#include <boost/utility.hpp>

#include <stdexcept>
#include <type_traits>

namespace alps {
    namespace accumulators {
        // this should be called namespace tag { struct mean; }
        // but gcc <= 4.4 has lookup error, so name it different
        struct mean_tag;

        template<typename T> struct mean_type
            : public std::conditional<std::is_integral<typename value_type<T>::type>::value, double, typename value_type<T>::type>
        {};

        template<typename T> struct has_feature<T, mean_tag> {
            template<typename R, typename C> static char helper(R(C::*)() const);
            template<typename C> static char check(std::integral_constant<std::size_t, sizeof(helper(&C::mean))>*);
            template<typename C> static double check(...);
            typedef std::integral_constant<bool, sizeof(char) == sizeof(check<T>(0))> type;
            constexpr static bool value = type::value;
        };

        template<typename T> typename mean_type<T>::type mean(T const & arg) {
            return arg.mean();
        }

        namespace detail {

            template<typename A> typename std::enable_if<
                  has_feature<A, mean_tag>::value
                , typename mean_type<A>::type
            >::type mean_impl(A const & acc) {
                return mean(acc);
            }

            template<typename A> typename std::enable_if<
                  !has_feature<A, mean_tag>::value
                , typename mean_type<A>::type
            >::type mean_impl(A const & acc) {
                throw std::runtime_error(std::string(typeid(A).name()) + " has no mean-method" + ALPS_STACKTRACE);
                return typename mean_type<A>::type();
            }
        }

        namespace impl {

            template<typename T, typename B> struct Accumulator<T, mean_tag, B> : public B {

                public:
                    typedef typename alps::accumulators::mean_type<B>::type mean_type;
                    typedef Result<T, mean_tag, typename B::result_type> result_type;

                    Accumulator(): B(), m_sum(T()) {}
                    Accumulator(Accumulator const & arg): B(arg), m_sum(arg.m_sum) {}

                    template<typename ArgumentPack> Accumulator(ArgumentPack const & args, typename std::enable_if<!is_accumulator<ArgumentPack>::value, int>::type = 0)
                        : B(args), m_sum(T())
                    {}

                    mean_type const mean() const;

                    using B::operator();
                    void operator()(T const & val);

                    template<typename S> void print(S & os, bool terse=false) const {
                        os << alps::short_print(mean());
                        B::print(os, terse);
                    }

                    void save(hdf5::archive & ar) const;
                    void load(hdf5::archive & ar);

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar);

                    void reset() {
                        B::reset();
                        m_sum = T();
                    }

              /// Merge the sum (mean) of  given accumulator of type A into this sum (mean) @param rhs Accumulator to merge
              template <typename A>
              void merge(const A& rhs)
              {
                using alps::numeric::operator+=;
                using alps::numeric::check_size;
                B::merge(rhs);
                check_size(m_sum,rhs.m_sum);
                m_sum += rhs.m_sum;
              }

#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          alps::mpi::communicator const & comm
                        , int root
                    );
                    void collective_merge(
                          alps::mpi::communicator const & comm
                        , int root
                    ) const;
#endif
                protected:

                    T const & sum() const;

                private:
                    T m_sum;
            };

            template<typename T, typename B> class Result<T, mean_tag, B> : public B {

                public:
                    typedef typename alps::accumulators::mean_type<B>::type mean_type;
                    typedef typename detail::make_scalar_result_type<impl::Result,T,mean_tag,B>::type scalar_result_type;

                    Result()
                        : B()
                        , m_mean(mean_type())
                    {}

                    template<typename A> Result(A const & acc)
                        : B(acc)
                        , m_mean(detail::mean_impl(acc))
                    {}

                    mean_type const mean() const {
                        return m_mean;
                    }

                    template<typename S> void print(S & os, bool terse=false) const {
                        os << alps::short_print(mean());
                        B::print(os, terse);
                    }

                    void save(hdf5::archive & ar) const;
                    void load(hdf5::archive & ar);

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar);

                    template<typename U> void operator+=(U const & arg) { augadd(arg); }
                    template<typename U> void operator-=(U const & arg) { augsub(arg); }
                    template<typename U> void operator*=(U const & arg) { augmul(arg); }
                    template<typename U> void operator/=(U const & arg) { augdiv(arg); }
                    void negate();
                    void inverse();

                    #define NUMERIC_FUNCTION_DECLARATION(FUNCTION_NAME)              \
                        void FUNCTION_NAME ();

                    NUMERIC_FUNCTION_DECLARATION(sin)
                    NUMERIC_FUNCTION_DECLARATION(cos)
                    NUMERIC_FUNCTION_DECLARATION(tan)
                    NUMERIC_FUNCTION_DECLARATION(sinh)
                    NUMERIC_FUNCTION_DECLARATION(cosh)
                    NUMERIC_FUNCTION_DECLARATION(tanh)
                    NUMERIC_FUNCTION_DECLARATION(asin)
                    NUMERIC_FUNCTION_DECLARATION(acos)
                    NUMERIC_FUNCTION_DECLARATION(atan)
                    NUMERIC_FUNCTION_DECLARATION(abs)
                    NUMERIC_FUNCTION_DECLARATION(sqrt)
                    NUMERIC_FUNCTION_DECLARATION(log)
                    NUMERIC_FUNCTION_DECLARATION(sq)
                    NUMERIC_FUNCTION_DECLARATION(cb)
                    NUMERIC_FUNCTION_DECLARATION(cbrt)

                    #undef NUMERIC_FUNCTION_DECLARATION

                private:

                    mean_type m_mean;

                    #define NUMERIC_FUNCTION_OPERATOR(OP_NAME, OPEQ_NAME, OP, OP_TOKEN)                                                                                         \
                        template<typename U> void aug ## OP_TOKEN (U const & arg, typename std::enable_if<!std::is_scalar<U>::value, int>::type = 0) {                             \
                            using alps::numeric:: OP_NAME ;                                                                                                                     \
                            m_mean = m_mean OP arg.mean();                                                                                                                      \
                            B:: OPEQ_NAME (arg);                                                                                                                                \
                        }                                                                                                                                                       \
                       template<typename U> void aug ## OP_TOKEN (U const & arg,                                                                                                \
                                                                  typename std::enable_if<std::is_scalar<U>::value, int>::type = 0) {                                              \
                            using alps::numeric:: OP_NAME ;                                                                                                                     \
                            m_mean = m_mean OP static_cast<typename alps::numeric::scalar<mean_type>::type>(arg); \
                            B:: OPEQ_NAME (arg);                                                                                                                                \
                        }

                    NUMERIC_FUNCTION_OPERATOR(operator+, operator+=, +, add)
                    NUMERIC_FUNCTION_OPERATOR(operator-, operator-=, -, sub)
                    NUMERIC_FUNCTION_OPERATOR(operator*, operator*=, *, mul)
                    NUMERIC_FUNCTION_OPERATOR(operator/, operator/=, /, div)

                    #undef NUMERIC_FUNCTION_OPERATOR
            };

            template<typename T, typename B> class BaseWrapper<T, mean_tag, B> : public B {
                public:
                    virtual bool has_mean() const = 0;
                    virtual typename mean_type<B>::type mean() const = 0;
            };

            template<typename T, typename B> class DerivedWrapper<T, mean_tag, B> : public B {
                public:
                    DerivedWrapper(): B() {}
                    DerivedWrapper(T const & arg): B(arg) {}

                    bool has_mean() const { return has_feature<T, mean_tag>::type::value; }

                    typename mean_type<B>::type mean() const { return detail::mean_impl(this->m_data); }
            };

        }
    }
}
