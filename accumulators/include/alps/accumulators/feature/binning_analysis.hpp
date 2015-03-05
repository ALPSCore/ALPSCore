/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_ACCUMULATOR_BINNING_ANALYSIS_HPP
#define ALPS_ACCUMULATOR_BINNING_ANALYSIS_HPP

#include <alps/config.hpp>

#include <alps/accumulators/feature.hpp>
#include <alps/accumulators/parameter.hpp>
#include <alps/accumulators/feature/mean.hpp>
#include <alps/accumulators/feature/count.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/short_print.hpp>

#include <alps/accumulators/convergence.hpp>
#include <alps/numeric/inf.hpp>
#include <alps/numeric/check_size.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/numeric/boost_array_functions.hpp>
#include <alps/numeric/set_negative_0.hpp>
// TODO: make nicer way to use this
#include <alps/type_traits/slice.hpp>
#include <alps/type_traits/change_value_type.hpp>

#include <boost/mpl/if.hpp>
#include <boost/utility.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/is_integral.hpp>

#include <limits>
#include <stdexcept>

// DEBUG: to force boost assertion to be an exception, to work nicely with google test
#define BOOST_ENABLE_ASSERT_HANDLER
#include "boost/assert.hpp"

namespace boost {
    inline void assertion_failed_msg(char const * expr, char const * msg, char const * function, char const * file, long line)
    {
        std::ostringstream errmsg;
        errmsg << "Boost assertion " << expr << " failed in "
               << function << ":\n"
               << file << "(" << line << "): "
               << msg;
        throw std::logic_error(errmsg.str());
    }

    inline void assertion_failed(char const * expr, char const * function, char const * file, long line)
    {
        std::ostringstream errmsg;
        errmsg << "Boost assertion " << expr << " failed in "
               << function << ":\n"
               << file << "(" << line << ")";
        throw std::logic_error(errmsg.str());
    }
}

namespace alps {
    namespace accumulators {
        // this should be called namespace tag { struct binning_analysis; }
        // but gcc <= 4.4 has lookup error, so name it different
        struct binning_analysis_tag;

        template<typename T> struct autocorrelation_type
            : public boost::mpl::if_<boost::is_integral<typename value_type<T>::type>, double, typename value_type<T>::type>
        {};

        template<typename T> struct convergence_type {
            typedef typename change_value_type<typename value_type<T>::type, int>::type type;
        };

        template<typename T> struct has_feature<T, binning_analysis_tag> {
            template<typename R, typename C> static char helper(R(C::*)() const);
            template<typename C> static char check(boost::integral_constant<std::size_t, sizeof(helper(&C::autocorrelation))>*);
            template<typename C> static double check(...);
            typedef boost::integral_constant<bool, sizeof(char) == sizeof(check<T>(0))> type;
        };

        template<typename T> typename autocorrelation_type<T>::type autocorrelation(T const & arg) {
            return arg.autocorrelation();
        }

        namespace detail {

            template<typename A> typename boost::enable_if<
                  typename has_feature<A, binning_analysis_tag>::type
                , typename autocorrelation_type<A>::type
            >::type autocorrelation_impl(A const & acc) {
                return autocorrelation(acc);
            }

            template<typename A> typename boost::disable_if<
                  typename has_feature<A, binning_analysis_tag>::type
                , typename autocorrelation_type<A>::type
            >::type autocorrelation_impl(A const & acc) {
                throw std::runtime_error(std::string(typeid(A).name()) + " has no autocorrelation-method" + ALPS_STACKTRACE);
                return *static_cast<typename autocorrelation_type<A>::type *>(NULL);
            }
        }

        namespace impl {

            template<typename T, typename B> struct Accumulator<T, binning_analysis_tag, B> : public B {

                public:
                    typedef Result<T, binning_analysis_tag, typename B::result_type> result_type;

                    Accumulator()
                        : B()
                        , m_ac_sum()
                        , m_ac_sum2()
                        , m_ac_partial()
                        , m_ac_count()
                    {}

                    Accumulator(Accumulator const & arg)
                        : B(arg)
                        , m_ac_sum(arg.m_ac_sum)
                        , m_ac_sum2(arg.m_ac_sum2)
                        , m_ac_partial(arg.m_ac_partial)
                        , m_ac_count(arg.m_ac_count)
                    {}

                    template<typename ArgumentPack> Accumulator(ArgumentPack const & args, typename boost::disable_if<is_accumulator<ArgumentPack>, int>::type = 0)
                        : B(args)
                        , m_ac_sum()
                        , m_ac_sum2()
                        , m_ac_partial()
                        , m_ac_count()
                    {}                    

                    typename alps::accumulators::convergence_type<B>::type converged_errors() const {
                        typedef typename alps::hdf5::scalar_type<typename convergence_type<T>::type>::type convergence_scalar_type;

                        typename alps::accumulators::convergence_type<B>::type conv;
                        typename alps::accumulators::error_type<B>::type err = error();
                        check_size(conv, err);
                        const unsigned int range = 4;
                        const unsigned int depth = (m_ac_sum2.size() < 8 ? 1 : m_ac_sum2.size() - 7);
                        if (depth < range)
                            conv = 0 * conv + (convergence_scalar_type)MAYBE_CONVERGED;
                        else {
                            conv = 0 * conv + (convergence_scalar_type)CONVERGED;
                            // TODO: how to we iterate over the datatype?
                            for (unsigned int i = depth - range; i < depth - 1; ++i) {
                                typename slice_index<typename alps::accumulators::convergence_type<B>::type>::type it;
                                result_type this_err(error(i));
                                for (it = slices(conv).first; it != slices(conv).second; ++it)
                                    if (std::abs(slice_value(this_err, it)) >= std::abs(slice_value(err,it)))
                                        slice_value(conv,it) = CONVERGED;
                                    else if (std::abs(slice_value(this_err, it)) < 0.824 * std::abs(slice_value(err,it)))
                                        slice_value(conv,it) = NOT_CONVERGED;
                                    else if (std::abs(slice_value(this_err, it)) < 0.9 * std::abs(slice_value(err,it)) && slice_value(conv, it) != NOT_CONVERGED)
                                        slice_value(conv,it) = MAYBE_CONVERGED;
                            }
                        }
                        return conv;
                    }

                    typename alps::accumulators::error_type<B>::type const error(std::size_t bin_level = std::numeric_limits<std::size_t>::max()) const {
                        using alps::numeric::operator*;
                        using alps::numeric::operator-;
                        using alps::numeric::operator/;
                        using std::sqrt;
                        using alps::numeric::sqrt;

                        // FIXME: here and in other places there are magic numbers: 8, 7 (presumably 8-1) and 4 (presumably 8/2).
                        if (m_ac_sum2.size()<8) {
                            bin_level = 0;
                        } else if (bin_level > m_ac_sum2.size() - 8) {
                            bin_level = m_ac_sum2.size() - 8;
                        }

                        typedef typename alps::accumulators::error_type<B>::type error_type;
                        typedef typename alps::hdf5::scalar_type<error_type>::type error_scalar_type;

                        // if not enough bins are available, return infinity
                        if (m_ac_sum2.size() < 2)
                            return alps::numeric::inf<error_type>();

                        // TODO: make library for scalar type
                        error_scalar_type one = 1;

                        error_scalar_type binlen = 1ll << bin_level;
                        BOOST_ASSERT_MSG(bin_level<m_ac_count.size(),"bin_level within the range of m_ac_count");
                        error_scalar_type N_i = m_ac_count[bin_level];
                        BOOST_ASSERT_MSG(bin_level<m_ac_sum.size(),"bin_level within the range of m_ac_sum");
                        error_type sum_i = m_ac_sum[bin_level];
                        BOOST_ASSERT_MSG(bin_level<m_ac_sum2.size(),"bin_level within the range of m_ac_sum2");
                        error_type sum2_i = m_ac_sum2[bin_level];
                        error_type var_i = (sum2_i / binlen - sum_i * sum_i / (N_i * binlen)) / (N_i * binlen);
                        return sqrt(var_i / (N_i - one));
                    }

                    typename autocorrelation_type<B>::type const autocorrelation() const {
                        using alps::numeric::operator*;
                        using alps::numeric::operator-;
                        using alps::numeric::operator/;

                        typedef typename mean_type<B>::type mean_type;

                        // TODO: make library for scalar type
                        typedef typename alps::hdf5::scalar_type<mean_type>::type mean_scalar_type;

                        // if not enoght bins are available, return infinity
                        if (m_ac_sum2.size() < 2)
                            return alps::numeric::inf<mean_type>();

                        mean_scalar_type one = 1;
                        mean_scalar_type two = 2;

                        mean_scalar_type N_0 = m_ac_count[0];
                        mean_type sum_0 = m_ac_sum[0];
                        mean_type sum2_0 = m_ac_sum2[0];
                        mean_type var_0 = (sum2_0 - sum_0 * sum_0 / N_0) / N_0;
                        alps::numeric::set_negative_0(var_0);
                        mean_scalar_type fac = B::count() - 1;
                        mean_type err = error();
                        return (err * err * fac / var_0 - one) / two;
                    }

                    uint32_t binning_depth() const {
                        return m_ac_sum2.size() < 8 ? 1 : m_ac_sum2.size() - 7;
                    }

                    using B::operator();
                    void operator()(T const & val) {
                        using alps::numeric::operator+=;
                        using alps::numeric::operator*;
                        using alps::numeric::check_size;

                        B::operator()(val);
                        if(B::count() == (1 << m_ac_sum2.size())) {
                            m_ac_sum2.push_back(T());
                            check_size(m_ac_sum2.back(), val);
                            m_ac_sum.push_back(T());
                            check_size(m_ac_sum.back(), val);
                            m_ac_partial.push_back(m_ac_sum[0]);
                            check_size(m_ac_partial.back(), val);
                            m_ac_count.push_back(typename count_type<B>::type());
                        }
                        BOOST_ASSERT_MSG(m_ac_partial.size() >= m_ac_sum2.size(), "m_ac_partial is as large as m_ac_sum2");
                        BOOST_ASSERT_MSG(m_ac_count.size() >= m_ac_sum2.size(), "m_ac_count is as large as m_ac_sum2");
                        BOOST_ASSERT_MSG(m_ac_sum.size() >= m_ac_sum2.size(), "m_ac_sum is as large as m_ac_sum2");
                        for (unsigned i = 0; i < m_ac_sum2.size(); ++i) {
                            m_ac_partial[i] += val;
                            if (!(B::count() & ((1ll << i) - 1))) {
                                m_ac_sum2[i] += m_ac_partial[i] * m_ac_partial[i];
                                m_ac_sum[i] += m_ac_partial[i];
                                m_ac_count[i]++;
                                m_ac_partial[i] = T();
                                check_size(m_ac_partial[i], val);
                            }                            
                        }
                    }

                    template<typename S> void print(S & os) const {
                        B::print(os);
                        os << " Autocorrelation: " << short_print(autocorrelation());
                        if (m_ac_sum2.size() > 0) {
                            for (std::size_t i = 0; i < binning_depth(); ++i)
                                os << std::endl
                                    << "    bin #" << std::setw(3) <<  i + 1
                                    << " : " << std::setw(8) << m_ac_count[i]
                                    << " entries: error = " << short_print(error(i));
                            os << std::endl;
                        } else
                            os << "No mesurements" << std::endl;
                    }

                    void save(hdf5::archive & ar) const {
                        B::save(ar);
                        if (B::count())
                            ar["tau/partialbin"] = m_ac_sum;
                        ar["tau/data"] = m_ac_sum2;
                        ar["tau/ac_count"] = m_ac_count; // FIXME: proper dataset name? to be saved always?
                        ar["tau/ac_partial"] = m_ac_partial;  // FIXME: proper dataset name? to be saved always?
                    }

                    void load(hdf5::archive & ar) { // TODO: make archive const
                        B::load(ar);
                        if (ar.is_data("tau/partialbin"))
                            ar["tau/partialbin"] >> m_ac_sum;
                        ar["tau/data"] >> m_ac_sum2;
                        if (ar.is_data("tau/ac_count"))
                            ar["tau/ac_count"] >> m_ac_count; // FIXME: proper dataset name?
                        if (ar.is_data("tau/ac_partial"))
                            ar["tau/ac_partial"] >> m_ac_partial;  // FIXME: proper dataset name?
                    }

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::hdf5::get_extent;

                        return B::can_load(ar)
                            && ar.is_data("tau/data")
                            && get_extent(T()).size() + 1 == ar.dimensions("tau/data")
                        ;
                    }

                    void reset() {
                        B::reset();
                        m_ac_sum = std::vector<T>();
                        m_ac_sum2 = std::vector<T>();
                        m_ac_partial = std::vector<T>();
                        m_ac_count = std::vector<typename count_type<B>::type>();
                    }

                   /// Merge placeholder \remark FIXME: always throws
                    template <typename A>
                    void merge(const A& rhs)
                    {
                      throw std::logic_error("Merging binning accumulators is not yet implemented"
                                             + ALPS_STACKTRACE);
                    }
#ifdef ALPS_HAVE_MPI
                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) {

                        if (comm.rank() == root) {
                            B::collective_merge(comm, root);
                            typedef typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type mean_scalar_type;
                            std::size_t size = boost::mpi::all_reduce(comm, m_ac_count.size(), boost::mpi::maximum<std::size_t>());

                            m_ac_count.resize(size);
                            B::reduce_if(comm, std::vector<typename count_type<B>::type>(m_ac_count), m_ac_count, std::plus<typename count_type<B>::type>(), root);

                            m_ac_sum.resize(size);
                            B::reduce_if(comm, std::vector<T>(m_ac_sum), m_ac_sum, std::plus<mean_scalar_type>(), root);

                            m_ac_sum2.resize(size);
                            B::reduce_if(comm, std::vector<T>(m_ac_sum2), m_ac_sum2, std::plus<mean_scalar_type>(), root);

                        } else
                            const_cast<Accumulator<T, binning_analysis_tag, B> const *>(this)->collective_merge(comm, root);
                    }

                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
                        B::collective_merge(comm, root);
                        if (comm.rank() == root)
                            throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
                        else {
                            typedef typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type mean_scalar_type;

                            std::size_t size = boost::mpi::all_reduce(comm, m_ac_count.size(), boost::mpi::maximum<std::size_t>());
                            {
                                std::vector<typename count_type<B>::type> count(m_ac_count);
                                count.resize(size);
                                B::reduce_if(comm, count, std::plus<typename count_type<B>::type>(), root);
                            }
                            {
                                std::vector<T> sum(m_ac_sum);
                                sum.resize(size);
                                B::reduce_if(comm, sum, std::plus<mean_scalar_type>(), root);
                            }
                            {
                                std::vector<T> sum2(m_ac_sum2);
                                sum2.resize(size);
                                B::reduce_if(comm, sum2, std::plus<mean_scalar_type>(), root);
                            }
                        }
                    }
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
                typedef typename alps::hdf5::scalar_type<error_type>::type error_scalar_type;
                typedef typename std::vector<error_type>::iterator error_iterator;

                public:
                    typedef typename alps::accumulators::autocorrelation_type<B>::type autocorrelation_type;

                    Result()
                        : B()
                        , m_ac_autocorrelation()
                        , m_ac_errors()
                    {}

                    template<typename A> Result(A const & acc)
                        : B(acc)
                        , m_ac_autocorrelation(detail::autocorrelation_impl(acc))
                        , m_ac_errors(acc.binning_depth())
                    {
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = acc.error(it - m_ac_errors.begin());
                    }

                    error_type const error(std::size_t bin_level = std::numeric_limits<std::size_t>::max()) const {
                        if (m_ac_errors.size() < 2)
                            return alps::numeric::inf<error_type>();
                        return m_ac_errors[bin_level >= m_ac_errors.size() ? 0 : bin_level];
                    }

                    autocorrelation_type const autocorrelation() const {
                        return m_ac_autocorrelation;
                    }

                    template<typename S> void print(S & os) const {
                        B::print(os);
                        os << " Tau: " << short_print(autocorrelation());
                        if (m_ac_errors.size() > 0) {
                            for (std::size_t i = 0; i < m_ac_errors.size(); ++i)
                                os << std::endl
                                    << "    bin #" << std::setw(3) <<  i + 1
                                    << " entries: error = " << short_print(m_ac_errors[i]);
                            os << std::endl;
                        }
                    }

                    void save(hdf5::archive & ar) const {
                        B::save(ar);
                        ar["tau"] = m_ac_autocorrelation;
                    }

                    void load(hdf5::archive & ar) {
                        B::load(ar);
                        ar["tau"] >> m_ac_autocorrelation;
                    }

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::hdf5::get_extent;

                        return B::can_load(ar)
                            && ar.is_data("tau")
                            && get_extent(T()).size() + 1 == ar.dimensions("tau")
                        ;
                    }

                    template<typename U> void operator+=(U const & arg) { augaddsub(arg); B::operator+=(arg); }
                    template<typename U> void operator-=(U const & arg) { augaddsub(arg); B::operator-=(arg); }
                    template<typename U> void operator*=(U const & arg) { augmul(arg); }
                    template<typename U> void operator/=(U const & arg) { augdiv(arg); }
                    void negate() {
                        using alps::numeric::operator-;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = -*it;
                        B::negate();
                    }
                    void inverse() {
                        using alps::numeric::operator*;
                        using alps::numeric::operator/;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = dynamic_cast<self_type &>(*this).error(it - m_ac_errors.begin()) / (this->mean() * this->mean());
                        B::inverse();
                    }

                    #define NUMERIC_FUNCTION_USING                                  \
                        using alps::numeric::sq;                                    \
                        using alps::numeric::cbrt;                                  \
                        using alps::numeric::cb;                                    \
                        using std::sqrt;                                            \
                        using alps::numeric::sqrt;                                  \
                        using std::exp;                                             \
                        using alps::numeric::exp;                                   \
                        using std::log;                                             \
                        using alps::numeric::log;                                   \
                        using std::abs;                                             \
                        using alps::numeric::abs;                                   \
                        using std::pow;                                             \
                        using alps::numeric::pow;                                   \
                        using std::sin;                                             \
                        using alps::numeric::sin;                                   \
                        using std::cos;                                             \
                        using alps::numeric::cos;                                   \
                        using std::tan;                                             \
                        using alps::numeric::tan;                                   \
                        using std::sinh;                                            \
                        using alps::numeric::sinh;                                  \
                        using std::cosh;                                            \
                        using alps::numeric::cosh;                                  \
                        using std::tanh;                                            \
                        using alps::numeric::tanh;                                  \
                        using std::asin;                                            \
                        using alps::numeric::asin;                                  \
                        using std::acos;                                            \
                        using alps::numeric::acos;                                  \
                        using std::atan;                                            \
                        using alps::numeric::atan;                                  \
                        using alps::numeric::operator+;                             \
                        using alps::numeric::operator-;                             \
                        using alps::numeric::operator*;                             \
                        using alps::numeric::operator/;

                    void sin() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(cos(this->mean()) * *it);
                    }

                    void cos() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(-sin(this->mean()) * *it);
                    }

                    void tan() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(error_scalar_type(1) / (cos(this->mean()) * cos(this->mean())) * *it);
                    }

                    void sinh() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(cosh(this->mean()) * *it);
                    }

                    void cosh() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(sinh(this->mean()) * *it);
                    }

                    void tanh() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(error_scalar_type(1) / (cosh(this->mean()) * cosh(this->mean())) * *it);
                    }

                    void asin() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(error_scalar_type(1) / sqrt(error_scalar_type(1) - this->mean() * this->mean()) * *it);
                    }

                    void acos() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(error_scalar_type(-1) / sqrt(error_scalar_type(1) - this->mean() * this->mean()) * *it);
                    }

                    void atan() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(error_scalar_type(1) / (error_scalar_type(1) + this->mean() * this->mean()) * *it);
                    }

                    // abs does not change the error, so nothing has to be done ...

                    void sq() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(error_scalar_type(2) * this->mean() * *it);
                    }

                    void sqrt() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(*it / (error_scalar_type(2) * sqrt(this->mean())));
                    }

                    void cb() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(error_scalar_type(3) * sq(this->mean()) * *it);
                    }

                    void cbrt() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(*it / (error_scalar_type(3) * sq(pow(this->mean(), error_scalar_type(1./3.)))));
                    }

                    void exp() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = exp(this->mean()) * *it;
                    }

                    void log() {
                        B::sin();
                        NUMERIC_FUNCTION_USING
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = abs(*it / this->mean());
                    }

                private:
                    typename mean_type<B>::type m_ac_autocorrelation;
                    std::vector<typename alps::accumulators::error_type<B>::type> m_ac_errors;

                    template<typename U> void augaddsub (U const & arg, typename boost::disable_if<boost::is_scalar<U>, int>::type = 0) {
                        using alps::numeric::operator+;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = *it + dynamic_cast<self_type const &>(arg).error(it - m_ac_errors.begin());
                    }
                    template<typename U> void augaddsub (U const & arg, typename boost::enable_if<boost::mpl::and_<
                          boost::is_scalar<U>
                        , typename has_operator_add<error_type, U>::type
                    >, int>::type = 0) {}
                    template<typename U> void augaddsub (U const & arg, typename boost::enable_if<boost::mpl::and_<
                          boost::is_scalar<U>
                        , boost::mpl::not_<typename has_operator_add<error_type, U>::type>
                    >, int>::type = 0) {
                        throw std::runtime_error(std::string(typeid(error_type).name()) + " has no operator + " + typeid(U).name() + ALPS_STACKTRACE);
                    }

                    template<typename U> void augmul (U const & arg, typename boost::disable_if<boost::is_scalar<U>, int>::type = 0) {
                        using alps::numeric::operator*;
                        using alps::numeric::operator+;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = arg.mean() * *it + this->mean() * dynamic_cast<self_type const &>(arg).error(it - m_ac_errors.begin());
                        B::operator*=(arg);
                    }
                    template<typename U> void augmul (U const & arg, typename boost::enable_if<boost::mpl::and_<
                          boost::is_scalar<U>
                        , typename has_operator_mul<error_type, U>::type
                    >, int>::type = 0) {
                        using alps::numeric::operator*;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = *it * arg;
                        B::operator*=(arg);
                    }
                    template<typename U> void augmul (U const & arg, typename boost::enable_if<boost::mpl::and_<
                          boost::is_scalar<U>
                        , boost::mpl::not_<typename has_operator_mul<error_type, U>::type>
                    >, int>::type = 0) {
                        throw std::runtime_error(std::string(typeid(error_type).name()) + " has no operator * " + typeid(U).name() + ALPS_STACKTRACE);
                    }

                    template<typename U> void augdiv (U const & arg, typename boost::disable_if<boost::is_scalar<U>, int>::type = 0) {
                        using alps::numeric::operator*;
                        using alps::numeric::operator/;
                        using alps::numeric::operator+;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = *it / arg.mean() + this->mean() * dynamic_cast<self_type const &>(arg).error(it - m_ac_errors.begin()) / (arg.mean() * arg.mean());
                        B::operator/=(arg);
                    }
                    template<typename U> void augdiv (U const & arg, typename boost::enable_if<boost::mpl::and_<
                          boost::is_scalar<U>
                        , typename has_operator_div<error_type, U>::type
                    >, int>::type = 0) {
                        using alps::numeric::operator/;
                        for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                            *it = *it / arg;
                        B::operator/=(arg);
                    }
                    template<typename U> void augdiv (U const & arg, typename boost::enable_if<boost::mpl::and_<
                          boost::is_scalar<U>
                        , boost::mpl::not_<typename has_operator_div<error_type, U>::type>
                    >, int>::type = 0) {
                        throw std::runtime_error(std::string(typeid(error_type).name()) + " has no operator / " + typeid(U).name() + ALPS_STACKTRACE);
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

 #endif
