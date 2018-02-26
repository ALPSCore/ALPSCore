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
#include <alps/accumulators/feature/error.hpp>

#include <alps/numeric/inf.hpp>
#include <alps/numeric/boost_array_functions.hpp>
#include <alps/numeric/check_size.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/short_print.hpp>

#include <alps/numeric/outer_product.hpp>
#include <alps/type_traits/covariance_type.hpp>

#include <boost/utility.hpp>
#include <boost/function.hpp>

#include <stdexcept>
#include <type_traits>

namespace alps {
    namespace accumulators {
        // this should be called namespace tag { struct max_num_binning; }
        // but gcc <= 4.4 has lookup error, so name it different
        struct max_num_binning_tag;

        namespace detail {
            template<typename C, typename M> class max_num_binning_proxy {
                typedef typename std::size_t size_type;

              public:
                max_num_binning_proxy(std::vector<M> const & bins, C const & num_elements, size_type const & max_number)
                    : m_max_number(max_number)
                    , m_num_elements(num_elements)
                    , m_bins(bins)
                {}

                std::vector<M> const & bins() const {
                    return m_bins;
                }

                C num_elements() const {
                    return m_num_elements;
                }

                size_type max_number() const {
                    return m_max_number;
                }

                std::ostream& print(std::ostream& os, bool terse) const
                {
                    if (m_bins.empty()) {
                        os << "No Bins";
                        return os;
                    }
                    if (terse) {
                        os << alps::short_print(m_bins, 4) << "#" << m_num_elements;
                        return os;
                    }
                    os << m_num_elements << " elements per bin, bins are:\n";
                    for (size_t i=0; i<m_bins.size(); ++i) {
                        os << "#" << (i+1) << ": " << alps::short_print(m_bins[i],4) << "\n";
                    }
                    return os;
                }

                private:

                size_type m_max_number;
                C m_num_elements;
                std::vector<M> const & m_bins;
            };

            template<typename C, typename M> inline std::ostream & operator<<(std::ostream & os, max_num_binning_proxy<C, M> const & arg) {
                return arg.print(os,true);
            };
        }

        template<typename T> struct max_num_binning_type {
            typedef detail::max_num_binning_proxy<typename count_type<T>::type, typename mean_type<T>::type> type;
        };

        template<typename T> struct has_feature<T, max_num_binning_tag> {
            template<typename R, typename C> static char helper(R(C::*)() const);
            template<typename C> static char check(std::integral_constant<std::size_t, sizeof(helper(&C::max_num_binning))>*);
            template<typename C> static double check(...);
            typedef std::integral_constant<bool, sizeof(char) == sizeof(check<T>(0))> type;
            constexpr static bool value = type::value;
        };

        template<typename T> typename max_num_binning_type<T>::type max_num_binning(T const & arg) {
            return arg.max_num_binning();
        }

        template<typename T> struct covariance_type
            : public alps::covariance_type<typename value_type<T>::type>
        {};

        namespace detail {

            template<typename A> typename std::enable_if<
                  has_feature<A, max_num_binning_tag>::value
                , typename max_num_binning_type<A>::type
            >::type max_num_binning_impl(A const & acc) {
                return max_num_binning(acc);
            }
            template<typename A> typename std::enable_if<
                  !has_feature<A, max_num_binning_tag>::value
                , typename max_num_binning_type<A>::type
            >::type max_num_binning_impl(A const & /*acc*/) {
                throw std::runtime_error(std::string(typeid(A).name()) + " has no max_num_binning-method" + ALPS_STACKTRACE);
                return *static_cast<typename max_num_binning_type<A>::type *>(NULL);
            }

            template<typename A, typename OP> void transform_impl(
                  A & acc
                , OP op
                , typename std::enable_if<has_feature<A, max_num_binning_tag>::value, int>::type = 0
            ) {
                acc.transform(op);
            }
            template<typename A, typename OP> void transform_impl(
                  A & /*acc*/
                , OP /*op*/
                , typename std::enable_if<!has_feature<A, max_num_binning_tag>::value, int>::type = 0
            ) {
                throw std::runtime_error(std::string(typeid(A).name()) + " has no transform-method" + ALPS_STACKTRACE);
            }
        }

        namespace impl {

            template<typename T, typename B> struct Accumulator<T, max_num_binning_tag, B> : public B {

              public:
                typedef typename alps::accumulators::max_num_binning_type<B>::type max_num_binning_type;
                typedef Result<T, max_num_binning_tag, typename B::result_type> result_type;

                Accumulator();
                Accumulator(Accumulator const & arg);

                template<typename ArgumentPack> Accumulator(ArgumentPack const & args, typename std::enable_if<!is_accumulator<ArgumentPack>::value, int>::type = 0)
                    : B(args)
                    , m_mn_max_number(args[max_bin_number | 128])
                    , m_mn_elements_in_bin(0)
                    , m_mn_elements_in_partial(0)
                    , m_mn_partial(T())
                {}

                max_num_binning_type const max_num_binning() const {
                    return max_num_binning_type(m_mn_bins, m_mn_elements_in_bin, m_mn_max_number);
                }

                template <typename OP> void transform(OP) {
                    throw std::runtime_error("Transform can only be applied to a result" + ALPS_STACKTRACE);
                }

                template <typename U, typename OP> void transform(U const &, OP) {
                    throw std::runtime_error("Transform can only be applied to a result" + ALPS_STACKTRACE);
                }

                using B::operator();
                void operator()(T const & val);

                template<typename S> void print(S & os, bool terse=false) const {
                    if (terse) {
                        os << alps::short_print(this->mean())
                           << " #" << this->count()
                           << " +/-" << alps::short_print(this->error())
                           << " Tau:" << alps::short_print(this->autocorrelation());
                    } else {
                        B::print(os, terse);
                        os << "Full-binning accumulator state:\n"
                           << "Mean +/-error (tau): "
                           << alps::short_print(this->mean())
                           << " +/-" << alps::short_print(this->error())
                           << "(" << alps::short_print(this->autocorrelation()) << ")\n";
                        os << " Bins: ";
                        max_num_binning().print(os,false);
                    }
                }

                void save(hdf5::archive & ar) const;
                void load(hdf5::archive & ar);

                static std::size_t rank() { return B::rank() + 1; }

                static bool can_load(hdf5::archive & ar);

                void reset();

                /// Merge the bins of the given accumulator of type A into this accumulator @param rhs Accumulator to merge
                    /** @warning: FIXME: Incomplete, therefore always throws. */
                template <typename A>
                void merge(const A& rhs)
                {
                        throw std::logic_error("Merging of FullBinningAccumulators is not yet implemented.\n"
                                               "Please contact ALPSCore developers and provide the code\n"
                                               "where you are using the merge() method.");
                    B::merge(rhs);
                    // FIXME!!! Needs a test to proceed with the coding!
                }

#ifdef ALPS_HAVE_MPI
                void collective_merge(alps::mpi::communicator const & comm,
                                      int root);

                void collective_merge(alps::mpi::communicator const & comm,
                                      int root) const;

              private:
                void partition_bins(alps::mpi::communicator const & comm,
                                    std::vector<typename mean_type<B>::type> & local_bins,
                                    std::vector<typename mean_type<B>::type> & merged_bins,
                                    int /*root*/) const;
#endif

              private:

                std::size_t m_mn_max_number;
                typename B::count_type m_mn_elements_in_bin, m_mn_elements_in_partial;
                T m_mn_partial;
                std::vector<typename mean_type<B>::type> m_mn_bins;
            };



            template<typename T, typename B> class Result<T, max_num_binning_tag, B> : public B {

              public:
                typedef typename alps::accumulators::max_num_binning_type<B>::type max_num_binning_type;
                typedef typename detail::make_scalar_result_type<impl::Result,T,max_num_binning_tag,B>::type scalar_result_type;

                Result();

                // copy constructor
                template<typename A> Result(A const & acc, typename std::enable_if<std::is_base_of<ResultBase<T>, A>::value, int>::type = 0)
                    : B(acc)
                    , m_mn_max_number(acc.m_mn_max_number)
                    , m_mn_elements_in_bin(acc.m_mn_elements_in_bin)
                    , m_mn_bins(acc.m_mn_bins)
                    , m_mn_count(acc.count())
                    , m_mn_mean(acc.mean())
                    , m_mn_error(acc.error())
                    , m_mn_cannot_rebin(acc.m_mn_cannot_rebin)
                    , m_mn_jackknife_valid(acc.m_mn_jackknife_valid)
                    , m_mn_data_is_analyzed(acc.m_mn_data_is_analyzed)
                    , m_mn_jackknife_bins(acc.m_mn_jackknife_bins)
                {}

                template<typename A> Result(A const & acc, typename std::enable_if<!std::is_base_of<ResultBase<T>, A>::value, int>::type = 0)
                    : B(acc)
                    , m_mn_max_number(detail::max_num_binning_impl(acc).max_number())
                    , m_mn_elements_in_bin(detail::max_num_binning_impl(acc).num_elements())
                    , m_mn_bins(detail::max_num_binning_impl(acc).bins())
                    , m_mn_count(acc.count())
                    , m_mn_mean(acc.mean())
                    , m_mn_error(acc.error())
                    , m_mn_cannot_rebin(false)
                    , m_mn_jackknife_valid(false)
                    , m_mn_data_is_analyzed(true)
                    , m_mn_jackknife_bins(0)
                {}

                typename B::count_type count() const;
                typename mean_type<B>::type const & mean() const;
                typename error_type<B>::type const & error() const;

                max_num_binning_type const max_num_binning() const {
                    return max_num_binning_type(m_mn_bins, m_mn_elements_in_bin, m_mn_max_number);
                }

                // Seem to be broken ...
                /*
                template <typename A>
                typename std::enable_if<has_feature<A, max_num_binning_tag>::value,
                                          typename covariance_type<B>::type
                                         >::type covariance(A const & obs) const;

                // Adapted from http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance
                // It is a two-pass algorithm, which first calculates estimates for the mean and then performs
                // the stable algorithm on the residuals. According to literature and local authorities, this
                // is the most accurate and stable way to calculate variances.
                template <typename A> typename std::enable_if<
                    has_feature<A, max_num_binning_tag>::value, typename covariance_type<B>::type
                    >::type accurate_covariance(A const & obs) const;
                */

                // TODO: use mean error from here ...
                template<typename S> void print(S & os, bool terse=false) const {
                    // TODO: use m_mn_variables!
                    os << "Mean +/-error (tau): "
                       << alps::short_print(mean())
                       << " +/-" << alps::short_print(error())
                       << "(" << alps::short_print(this->autocorrelation()) << ")";
                    if (!terse) {
                        os << "\n Bins: " << max_num_binning();
                    }
                }

                void save(hdf5::archive & ar) const;
                void load(hdf5::archive & ar);

                static std::size_t rank() { return B::rank() + 1; }

                static bool can_load(hdf5::archive & ar);

                template<typename U> void operator+=(U const & arg) { augadd(arg); }
                template<typename U> void operator-=(U const & arg) { augsub(arg); }
                template<typename U> void operator*=(U const & arg) { augmul(arg); }
                template<typename U> void operator/=(U const & arg) { augdiv(arg); }

                template <typename OP> void transform(OP op) {
                    generate_jackknife();
                    m_mn_data_is_analyzed = false;
                    m_mn_cannot_rebin = true;
                    typename std::vector<typename mean_type<B>::type>::iterator it;
                    for (it = m_mn_bins.begin(); it != m_mn_bins.end(); ++it)
                        *it = op(*it);
                    for (it = m_mn_jackknife_bins.begin(); it != m_mn_jackknife_bins.end(); ++it)
                        *it = op(*it);
                    analyze();
                }

                template <typename OP, typename U> void transform(OP op, U const & arg) {
                    generate_jackknife();
                    arg.generate_jackknife(); /* TODO: make this more generic */
                    if (arg.get_jackknife_bins().size() != m_mn_jackknife_bins.size()) /* TODO: make this more generic */
                        throw std::runtime_error("Unable to transform: unequal number of bins" + ALPS_STACKTRACE);
                    m_mn_data_is_analyzed = false;
                    m_mn_cannot_rebin = true;
                    typename std::vector<typename mean_type<B>::type>::iterator it;
                    typename std::vector<typename mean_type<U>::type>::const_iterator jt;
                    for (it = m_mn_bins.begin(), jt = arg.get_bins().begin(); it != m_mn_bins.end(); ++it, ++jt)
                        *it = op(*it, *jt);
                    for (it = m_mn_jackknife_bins.begin(), jt = arg.get_jackknife_bins().begin(); it != m_mn_jackknife_bins.end(); ++it, ++jt)
                        *it = op(*it, *jt);
                }

                void sin();
                void cos();
                void tan();
                void sinh();
                void cosh();
                void tanh();
                void asin();
                void acos();
                void atan();
                void abs();
                void sq();
                void sqrt();
                void cb();
                void cbrt();
                void exp();
                void log();

                /// Negate the Result by calling transform() with the corresponding functor object
                void negate();

                /// Invert the Result by calling transform() with the corresponding functor object
                void inverse();

              private:
                std::size_t m_mn_max_number;
                typename B::count_type m_mn_elements_in_bin;
                std::vector<typename mean_type<B>::type> m_mn_bins;
                mutable typename B::count_type m_mn_count;
                mutable typename mean_type<B>::type m_mn_mean;
                mutable typename error_type<B>::type m_mn_error;
                mutable bool m_mn_cannot_rebin;
                mutable bool m_mn_jackknife_valid;
                mutable bool m_mn_data_is_analyzed;
                mutable std::vector<typename mean_type<B>::type> m_mn_jackknife_bins;

              public:
                const std::vector<typename mean_type<B>::type>& get_bins() const {
                    return m_mn_bins;
                }
                const std::vector<typename mean_type<B>::type>& get_jackknife_bins() const {
                    return m_mn_jackknife_bins;
                }
                void generate_jackknife() const;

              private:
                void analyze() const;

#define NUMERIC_FUNCTION_OPERATOR(OP_NAME, OPEQ_NAME, OP, OP_TOKEN, OP_STD) \
                template<typename U> void aug ## OP_TOKEN (U const & arg, typename std::enable_if<!std::is_scalar<U>::value, int>::type = 0) { \
                    typedef typename value_type<B>::type self_value_type; \
                    typedef typename value_type<U>::type arg_value_type; \
                    transform(boost::function<self_value_type(self_value_type, arg_value_type)>( OP_STD <self_value_type, arg_value_type, self_value_type>()), arg); \
                    B:: OPEQ_NAME (arg);                                \
                }                                                       \
                template<typename U> void aug ## OP_TOKEN (U const & arg, typename std::enable_if<std::is_scalar<U>::value, int>::type = 0) { \
                    using alps::numeric:: OP_NAME ;                     \
                    typedef typename mean_type<B>::type mean_type;      \
                    generate_jackknife();                               \
                    m_mn_data_is_analyzed = false;                      \
                    m_mn_cannot_rebin = true;                           \
                    typename std::vector<mean_type>::iterator it;       \
                    for (it = m_mn_bins.begin(); it != m_mn_bins.end(); ++it) \
                        *it = *it OP static_cast<typename alps::numeric::scalar<mean_type>::type>(arg); \
                    for (it = m_mn_jackknife_bins.begin(); it != m_mn_jackknife_bins.end(); ++it) \
                        *it = *it OP static_cast<typename alps::numeric::scalar<mean_type>::type>(arg); \
                    analyze();                                          \
                    B:: OPEQ_NAME (arg);                                \
                }                                                       \

                NUMERIC_FUNCTION_OPERATOR(operator+, operator+=, +, add, alps::numeric::plus)
                NUMERIC_FUNCTION_OPERATOR(operator-, operator-=, -, sub, alps::numeric::minus)
                NUMERIC_FUNCTION_OPERATOR(operator*, operator*=, *, mul, alps::numeric::multiplies)
                NUMERIC_FUNCTION_OPERATOR(operator/, operator/=, /, div, alps::numeric::divides)

#undef NUMERIC_FUNCTION_OPERATOR
            };

            template<typename T, typename B> class BaseWrapper<T, max_num_binning_tag, B> : public B {
              public:
                virtual bool has_max_num_binning() const = 0;
                virtual typename max_num_binning_type<B>::type max_num_binning() const = 0;
                virtual void transform(boost::function<typename value_type<B>::type(typename value_type<B>::type)>) = 0;
            };

            template<typename T, typename B> class DerivedWrapper<T, max_num_binning_tag, B> : public B {
                public:
                DerivedWrapper(): B() {}
                DerivedWrapper(T const & arg): B(arg) {}

                bool has_max_num_binning() const { return has_feature<T, max_num_binning_tag>::type::value; }
                bool has_transform() const { return has_feature<T, max_num_binning_tag>::type::value; }

                typename max_num_binning_type<B>::type max_num_binning() const { return detail::max_num_binning_impl(this->m_data); }
                void transform(boost::function<typename value_type<B>::type(typename value_type<B>::type)> op) { return detail::transform_impl(this->m_data, op); }
            };

        }
    }
}
