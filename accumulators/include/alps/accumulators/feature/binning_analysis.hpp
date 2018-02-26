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
#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/short_print.hpp>

#include <alps/numeric/inf.hpp>
#include <alps/numeric/check_size.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/numeric/boost_array_functions.hpp>
#include <alps/numeric/set_negative_0.hpp>
#include <alps/numeric/rectangularize.hpp>
// TODO: make nicer way to use this
#include <alps/type_traits/slice.hpp>
#include <alps/type_traits/change_value_type.hpp>

#include <boost/utility.hpp>

#include <limits>
#include <stdexcept>
#include <algorithm> // for std::min
#include <type_traits>

namespace alps {
    namespace accumulators {
        // this should be called namespace tag { struct binning_analysis; }
        // but gcc <= 4.4 has lookup error, so name it different
        struct binning_analysis_tag;

        template<typename T> struct autocorrelation_type
            : public std::conditional<std::is_integral<typename value_type<T>::type>::value, double, typename value_type<T>::type>
        {};

        template<typename T> struct convergence_type {
            typedef typename change_value_type<typename value_type<T>::type, int>::type type;
        };

        template<typename T> struct has_feature<T, binning_analysis_tag> {
            template<typename R, typename C> static char helper(R(C::*)() const);
            template<typename C> static char check(std::integral_constant<std::size_t, sizeof(helper(&C::autocorrelation))>*);
            template<typename C> static double check(...);
            typedef std::integral_constant<bool, sizeof(char) == sizeof(check<T>(0))> type;
            constexpr static bool value = type::value;
        };

        template<typename T> typename autocorrelation_type<T>::type autocorrelation(T const & arg) {
            return arg.autocorrelation();
        }

        namespace detail {

            template<typename A> typename std::enable_if<
                  has_feature<A, binning_analysis_tag>::value
                , typename autocorrelation_type<A>::type
            >::type autocorrelation_impl(A const & acc) {
                return autocorrelation(acc);
            }

            template<typename A> typename std::enable_if<
                  !has_feature<A, binning_analysis_tag>::value
                , typename autocorrelation_type<A>::type
            >::type autocorrelation_impl(A const & /*acc*/) {
                throw std::runtime_error(std::string(typeid(A).name()) + " has no autocorrelation-method" + ALPS_STACKTRACE);
                return *static_cast<typename autocorrelation_type<A>::type *>(NULL);
            }
        }

        namespace impl {

            template<typename T, typename B> struct Accumulator<T, binning_analysis_tag, B> : public B {

                public:
                    typedef Result<T, binning_analysis_tag, typename B::result_type> result_type;

                    Accumulator();
                    Accumulator(Accumulator const & arg);

                    template<typename ArgumentPack> Accumulator(ArgumentPack const & args, typename std::enable_if<!is_accumulator<ArgumentPack>::value, int>::type = 0)
                        : B(args)
                        , m_ac_sum()
                        , m_ac_sum2()
                        , m_ac_partial()
                        , m_ac_count()
                    {}

                    // This method is broken and does not compile
                    //typename alps::accumulators::convergence_type<B>::type converged_errors() const;

                    typename alps::accumulators::error_type<B>::type const error(std::size_t bin_level = std::numeric_limits<std::size_t>::max()) const;

                    uint32_t binning_depth() const {
                        return m_ac_sum2.size() < 8 ? 1 : m_ac_sum2.size() - 7;
                    }

                    typename autocorrelation_type<B>::type const autocorrelation() const;

                    using B::operator();
                    void operator()(T const & val);

                    template<typename S> void print(S & os, bool terse=false) const {
                        if (terse) {
                            os << alps::short_print(this->mean())
                               << " #" << this->count()
                               << " +/-" << alps::short_print(this->error())
                               << " Tau:" << alps::short_print(autocorrelation())
                               << " (warning: print result rather than accumulator)";
                        } else {
                            os << "DEBUG PRINTING of the accumulator object state (use mean(), error() and autocorrelation() methods instead)\n"
                               << "No-binning parent accumulator state:\n";
                            B::print(os, terse);
                            os << "\nLog-binning accumulator state:\n"
                               << " Error bar: " << alps::short_print(error())
                               << " Autocorrelation: " << alps::short_print(autocorrelation());
                            if (m_ac_sum2.size() > 0) {
                                for (std::size_t i = 0; i < binning_depth(); ++i)
                                    os << std::endl
                                       << "    bin #" << std::setw(3) <<  i + 1
                                       << " : " << std::setw(8) << m_ac_count[i]
                                       << " entries: error = " << alps::short_print(error(i));
                                os << std::endl;
                            } else
                                os << "No measurements" << std::endl;
                        }
                    }

                    void save(hdf5::archive & ar) const;
                    void load(hdf5::archive & ar);

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar);

                    void reset();

                    /// Merge the bins of the given accumulator of type A into this accumulator @param rhs Accumulator to merge
                    template <typename A>
                    void merge(const A& rhs)
                    {
                        using alps::numeric::merge;
                        B::merge(rhs);

                        merge(m_ac_count,rhs.m_ac_count);
                        merge(m_ac_sum,rhs.m_ac_sum);
                        merge(m_ac_sum2,rhs.m_ac_sum2);
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

                    std::vector<T> m_ac_sum;
                    std::vector<T> m_ac_sum2;
                    std::vector<T> m_ac_partial;
                    std::vector<typename count_type<B>::type> m_ac_count;
            };

            // TODO: remove autocorrelation on any transform
            template<typename T, typename B> class Result<T, binning_analysis_tag, B> : public B {

                typedef Result<T, binning_analysis_tag, B> self_type;
                typedef typename alps::accumulators::error_type<B>::type error_type;
                typedef typename alps::numeric::scalar<error_type>::type error_scalar_type;
                typedef typename std::vector<error_type>::iterator error_iterator;

                public:
                    typedef typename detail::make_scalar_result_type<impl::Result,T,binning_analysis_tag,B>::type scalar_result_type;
                    typedef typename alps::accumulators::autocorrelation_type<B>::type autocorrelation_type;

                    Result();

                    template<typename A> Result(A const & acc)
                        : B(acc)
                        , m_ac_autocorrelation(detail::autocorrelation_impl(acc))
                        , m_ac_errors(acc.binning_depth())
                    {
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = acc.error(it - m_ac_errors.begin());
                    }

                    error_type const error(std::size_t bin_level = std::numeric_limits<std::size_t>::max()) const;

                    autocorrelation_type const autocorrelation() const {
                        return m_ac_autocorrelation;
                    }

                    template<typename S> void print(S & os, bool terse=false) const {
                        if (terse) {
                            os << alps::short_print(this->mean())
                               << " #" << this->count()
                               << " +/-" << alps::short_print(this->error())
                               << " Tau:" << alps::short_print(autocorrelation());
                        } else {
                            os << " Error bar: " << alps::short_print(error());
                            os << " Autocorrelation: " << alps::short_print(autocorrelation());
                            if (m_ac_errors.size() > 0) {
                                for (std::size_t i = 0; i < m_ac_errors.size(); ++i)
                                    os << std::endl
                                       << "    bin #" << std::setw(3) <<  i + 1
                                       << " entries: error = " << alps::short_print(m_ac_errors[i]);
                            } else {
                                os << "No bins";
                            }
                            os << std::endl;
                        }
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

                    // abs does not change the error, so nothing has to be done ...

                    void sq();
                    void sqrt();
                    void cb();
                    void cbrt();
                    void exp();
                    void log();

                private:
                    typename mean_type<B>::type m_ac_autocorrelation;
                    std::vector<typename alps::accumulators::error_type<B>::type> m_ac_errors;

                    /// Predicate metafunction: is U is related to self_type?
                    template <typename U> struct is_rel_type
                        : public std::integral_constant<bool, std::is_same<U,self_type>::value ||
                                                              std::is_base_of<U,self_type>::value ||
                                                              std::is_base_of<self_type,U>::value > {};

                    /// Predicate metafunction: is U is related to self_type or is a scalar?
                    template <typename U> struct is_rel_or_scalar_type
                        : public std::integral_constant<bool, std::is_scalar<U>::value || is_rel_type<U>::value> {};


                    template<typename U> void augaddsub (U const & arg, typename std::enable_if<is_rel_type<U>::value, int>::type = 0) {
                        using alps::numeric::operator+;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = *it + dynamic_cast<self_type const &>(arg).error(it - m_ac_errors.begin());
                    }
                    template<typename U> void augaddsub (U const & arg, typename std::enable_if<!is_rel_or_scalar_type<U>::value, int>::type = 0) {
                        using alps::numeric::operator+;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = *it + dynamic_cast<scalar_result_type const &>(arg).error(it - m_ac_errors.begin());
                    }
                    template<typename U> void augaddsub (U const & /*arg*/, typename std::enable_if<std::is_scalar<U>::value, int>::type = 0) {}

                    template<typename U> void augmul (U const & arg, typename std::enable_if<is_rel_type<U>::value, int>::type = 0) {
                        using alps::numeric::operator*;
                        using alps::numeric::operator+;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = arg.mean() * *it + this->mean() * dynamic_cast<self_type const &>(arg).error(it - m_ac_errors.begin());
                        B::operator*=(arg);
                    }
                    template<typename U> void augmul (U const & arg, typename std::enable_if<!is_rel_or_scalar_type<U>::value, int>::type = 0) {
                        using alps::numeric::operator*;
                        using alps::numeric::operator+;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it =  (*it)*arg.mean() + this->mean() * dynamic_cast<scalar_result_type const &>(arg).error(it - m_ac_errors.begin());
                        B::operator*=(arg);
                    }
                    template<typename U> void augmul (U const & arg, typename std::enable_if<std::is_scalar<U>::value, int>::type = 0) {
                        using alps::numeric::operator*;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = *it * static_cast<error_scalar_type>(arg);
                        B::operator*=(arg);
                    }

                    template<typename U> void augdiv (U const & arg, typename std::enable_if<is_rel_type<U>::value, int>::type = 0) {
                        using alps::numeric::operator*;
                        using alps::numeric::operator/;
                        using alps::numeric::operator+;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = *it / arg.mean() + this->mean() * dynamic_cast<self_type const &>(arg).error(it - m_ac_errors.begin()) / (arg.mean() * arg.mean());
                        B::operator/=(arg);
                    }
                    template<typename U> void augdiv (U const & arg, typename std::enable_if<!is_rel_or_scalar_type<U>::value, int>::type = 0) {
                        using alps::numeric::operator*;
                        using alps::numeric::operator/;
                        using alps::numeric::operator+;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = *it / arg.mean() + this->mean() * dynamic_cast<scalar_result_type const &>(arg).error(it - m_ac_errors.begin()) / (arg.mean() * arg.mean());
                        B::operator/=(arg);
                    }
                    template<typename U> void augdiv (U const & arg, typename std::enable_if<std::is_scalar<U>::value, int>::type = 0) {
                        using alps::numeric::operator/;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = *it / static_cast<error_scalar_type>(arg);
                        B::operator/=(arg);
                    }
            };

            template<typename T, typename B> class BaseWrapper<T, binning_analysis_tag, B> : public B {
                public:
                    virtual bool has_autocorrelation() const = 0;
                    virtual typename autocorrelation_type<B>::type autocorrelation() const = 0;
            };

            template<typename T, typename B> class DerivedWrapper<T, binning_analysis_tag, B> : public B {
                public:
                    DerivedWrapper(): B() {}
                    DerivedWrapper(T const & arg): B(arg) {}

                    bool has_autocorrelation() const { return has_feature<T, binning_analysis_tag>::type::value; }

                    typename autocorrelation_type<B>::type autocorrelation() const { return detail::autocorrelation_impl(this->m_data); }
            };

        }
    }
}
