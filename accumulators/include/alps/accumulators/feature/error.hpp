/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>

#include <alps/accumulators/feature.hpp>
#include <alps/accumulators/parameter.hpp>
#include <alps/accumulators/feature/mean.hpp>
#include <alps/accumulators/feature/count.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/numeric/inf.hpp>
#include <alps/numeric/check_size.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/numeric/boost_array_functions.hpp>

#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/short_print.hpp>

#include <boost/utility.hpp>

#include <stdexcept>
#include <type_traits>

namespace alps {
    namespace accumulators {
        // this should be called namespace tag { struct error; }
        // but gcc <= 4.4 has lookup error, so name it different
        struct error_tag;

        template<typename T> struct error_type : public mean_type<T> {};

        template<typename T> struct has_feature<T, error_tag> {
            template<typename R, typename C> static char helper(R(C::*)() const);
            template<typename R, typename C> static char helper(R(C::*)(std::size_t) const);
            template<typename C> static char check(std::integral_constant<std::size_t, sizeof(helper(&C::error))>*);
            template<typename C> static double check(...);
            typedef std::integral_constant<bool, sizeof(char) == sizeof(check<T>(0))> type;
            constexpr static bool value = type::value;
        };

        template<typename T> typename error_type<T>::type error(T const & arg) {
            return arg.error();
        }

        namespace detail {

            template<typename A> typename std::enable_if<
                  has_feature<A, error_tag>::value
                , typename error_type<A>::type
            >::type error_impl(A const & acc) {
                return error(acc);
            }

            template<typename A> typename std::enable_if<
                  !has_feature<A, error_tag>::value
                , typename error_type<A>::type
            >::type error_impl(A const & /*acc*/) {
                throw std::runtime_error(std::string(typeid(A).name()) + " has no error-method" + ALPS_STACKTRACE);
                return typename error_type<A>::type();
            }
        }

        namespace impl {

            template<typename T, typename B> struct Accumulator<T, error_tag, B> : public B {

                public:
                    typedef typename alps::accumulators::error_type<B>::type error_type;
                    typedef typename alps::numeric::scalar<error_type>::type error_scalar_type;
                    typedef Result<T, error_tag, typename B::result_type> result_type;

                    Accumulator(): B(), m_sum2(T()) {}

                    Accumulator(Accumulator const & arg): B(arg), m_sum2(arg.m_sum2) {}

                    template<typename ArgumentPack> Accumulator(ArgumentPack const & args, typename std::enable_if<!is_accumulator<ArgumentPack>::value, int>::type = 0)
                        : B(args), m_sum2(T())
                    {}

                    error_type const error() const;

                    using B::operator();
                    void operator()(T const & val);

                    template<typename S> void print(S & os, bool terse=false) const {
                        B::print(os, terse);
                        os << " +/-" << alps::short_print(error());
                    }

                    void save(hdf5::archive & ar) const;
                    void load(hdf5::archive & ar);

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar);

                    void reset() {
                        B::reset();
                        m_sum2 = T();
                    }

                    /// Merge the mean & error of given accumulator of type A into this accumulator  @param rhs Accumulator to merge
                    template <typename A>
                    void merge(const A& rhs)
                    {
                      using alps::numeric::operator+=;
                      using alps::numeric::check_size;
                      B::merge(rhs);
                      check_size(m_sum2, rhs.m_sum2);
                      m_sum2 += rhs.m_sum2;
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

                private:
                    T m_sum2;
            };

            template<typename T, typename B> class Result<T, error_tag, B> : public B {

                public:
                    typedef typename alps::accumulators::error_type<B>::type error_type;
                    typedef typename alps::numeric::scalar<error_type>::type error_scalar_type; // FIXME: should be numeric::scalar<>
                    typedef typename detail::make_scalar_result_type<impl::Result,T,error_tag,B>::type scalar_result_type;

                    Result()
                        : B()
                        , m_error(error_type())
                    {}

                    template<typename A> Result(A const & acc)
                        : B(acc)
                        , m_error(detail::error_impl(acc))
                    {}

                    error_type const error() const {
                        return m_error;
                    }

                    template<typename S> void print(S & os, bool terse=false) const {
                        B::print(os, terse);
                        os << " +/-" << alps::short_print(error());
                    }

                    void save(hdf5::archive & ar) const;
                    void load(hdf5::archive & ar);

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar);

                    template<typename U> void operator+=(U const & arg) { augaddsub(arg); B::operator+=(arg); }
                    template<typename U> void operator-=(U const & arg) { augaddsub(arg); B::operator-=(arg); }
                    template<typename U> void operator*=(U const & arg) { augmul(arg); }
                    template<typename U> void operator/=(U const & arg) { augdiv(arg); }
                    void negate();
                    void inverse();

                    void sin();
                    void cos();
                    void tan();
                    void sinh();
                    void cosh();
                    void tanh();
                    void asin();
                    void acos();
                    void atan();
                    void sq();
                    void sqrt();
                    void cb();
                    void cbrt();
                    void exp();
                    void log();

                private:

                    error_type m_error;

                    template<typename U> void augaddsub (U const & arg, typename std::enable_if<!std::is_scalar<U>::value, int>::type = 0) {
                        using alps::numeric::operator+;
                        m_error = m_error + arg.error();
                    }
                    template<typename U> void augaddsub (U const & /*arg*/, typename std::enable_if<std::is_scalar<U>::value, int>::type = 0) {}

                    template<typename U> void augmul (U const & arg, typename std::enable_if<!std::is_scalar<U>::value, int>::type = 0) {
                        using alps::numeric::operator*;
                        using alps::numeric::operator+;
                        // FIXME? Originally: m_error = arg.mean() * m_error + this->mean() * arg.error();
                        // FIXME? Changed to:
                        m_error = m_error * arg.mean() + this->mean() * arg.error();
                        B::operator*=(arg);
                    }
                    template<typename U> void augmul (U const & arg, typename std::enable_if<std::is_scalar<U>::value, int>::type = 0) {
                        using alps::numeric::operator*;
                        m_error = m_error * static_cast<error_scalar_type>(arg);
                        B::operator*=(arg);
                    }

                    template<typename U> void augdiv (U const & arg, typename std::enable_if<!std::is_scalar<U>::value, int>::type = 0) {
                        using alps::numeric::operator*;
                        using alps::numeric::operator/;
                        using alps::numeric::operator+;
                        m_error = m_error / arg.mean() + this->mean() * arg.error() / (arg.mean() * arg.mean());
                        B::operator/=(arg);
                    }
                    template<typename U> void augdiv (U const & arg, typename std::enable_if<std::is_scalar<U>::value, int>::type = 0) {
                        using alps::numeric::operator/;
                        m_error = m_error / static_cast<error_scalar_type>(arg);
                        B::operator/=(arg);
                    }
            };

            template<typename T, typename B> class BaseWrapper<T, error_tag, B> : public B {
                public:
                    virtual bool has_error() const = 0;
                    virtual typename error_type<B>::type error() const = 0;
            };

            template<typename T, typename B> class DerivedWrapper<T, error_tag, B> : public B {
                public:
                    DerivedWrapper(): B() {}
                    DerivedWrapper(T const & arg): B(arg) {}

                    bool has_error() const { return has_feature<T, error_tag>::type::value; }

                    typename error_type<B>::type error() const { return detail::error_impl(this->m_data); }
            };

        }
    }
}
