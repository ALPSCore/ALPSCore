/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/accumulators/feature/error.hpp>
#include <alps/accumulators/feature/binning_analysis.hpp>
#include <alps/hdf5/vector.hpp>


#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#define ALPS_ACCUMULATOR_VALUE_TYPES_SEQ BOOST_PP_TUPLE_TO_SEQ(ALPS_ACCUMULATOR_VALUE_TYPES_SIZE, (ALPS_ACCUMULATOR_VALUE_TYPES))

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
        namespace impl {

            //
            // Accumulator<T, binning_analysis_tag, B>
            //

            template<typename T, typename B>
            Accumulator<T, binning_analysis_tag, B>::Accumulator()
                : B()
                , m_ac_sum()
                , m_ac_sum2()
                , m_ac_partial()
                , m_ac_count()
            {}

            template<typename T, typename B>
            Accumulator<T, binning_analysis_tag, B>::Accumulator(Accumulator const & arg)
                : B(arg)
                , m_ac_sum(arg.m_ac_sum)
                , m_ac_sum2(arg.m_ac_sum2)
                , m_ac_partial(arg.m_ac_partial)
                , m_ac_count(arg.m_ac_count)
            {}

            // This method is broken and does not compile
            // Apparently, it has never been instantiated before...
            /*
            template<typename T, typename B>
            typename alps::accumulators::convergence_type<B>::type Accumulator<T, binning_analysis_tag, B>::converged_errors() const {
                typedef typename alps::numeric::scalar<typename convergence_type<T>::type>::type convergence_scalar_type;

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
            }*/

            template<typename T, typename B>
            typename alps::accumulators::error_type<B>::type const
            Accumulator<T, binning_analysis_tag, B>::error(std::size_t bin_level) const {
                using alps::numeric::operator*;
                using alps::numeric::operator-;
                using alps::numeric::operator/;
                using std::sqrt;
                using alps::numeric::sqrt;

                // Basically, we try to estimate the integrated
                // autocorrelation time as
                //
                //           tau_int = s^2(n) / s^2(1),
                //
                // where s^2(n) is the sample variance when grouping
                // n measurements together in a bin.  In the above
                // approximation, there is a tradeoff to be had between
                //
                //  (1) formal validity, which improves with n,
                //  (2) statistical uncertainty, which improves with N/n,
                //
                // where N is the total number of steps. Here, 8 means
                // N/n = 2**8 = 256, which from the \chi^2 distribution
                // can be worked out to be a ~90% confidence interval
                // for an accuracy of 10%.
                if (m_ac_sum2.size()<8) {
                    bin_level = 0;
                } else if (bin_level > m_ac_sum2.size() - 8) {
                    bin_level = m_ac_sum2.size() - 8;
                }

                typedef typename alps::accumulators::error_type<B>::type error_type;
                typedef typename alps::numeric::scalar<error_type>::type error_scalar_type;

                // if not enough bins are available, return infinity
                if (m_ac_sum2.size() < 2)
                    return alps::numeric::inf<error_type>(B::error()); // FIXME: we call error() only to know the data size

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

            template<typename T, typename B>
            typename autocorrelation_type<B>::type const Accumulator<T, binning_analysis_tag, B>::autocorrelation() const {
                using alps::numeric::operator*;
                using alps::numeric::operator-;
                using alps::numeric::operator/;

                typedef typename mean_type<B>::type mean_type;

                // TODO: make library for scalar type
                typedef typename alps::numeric::scalar<mean_type>::type mean_scalar_type;

                mean_type err = error();

                // if not enoght bins are available, return infinity
                if (m_ac_sum2.size() < 2)
                    return alps::numeric::inf<mean_type>(err);

                mean_scalar_type one = 1;
                mean_scalar_type two = 2;

                mean_scalar_type N_0 = m_ac_count[0];
                mean_type sum_0 = m_ac_sum[0];
                mean_type sum2_0 = m_ac_sum2[0];
                mean_type var_0 = (sum2_0 - sum_0 * sum_0 / N_0) / N_0;
                alps::numeric::set_negative_0(var_0);
                mean_scalar_type fac = B::count() - 1;
                return (err * err * fac / var_0 - one) / two;
            }

            template<typename T, typename B>
            void Accumulator<T, binning_analysis_tag, B>::operator()(T const & val) {
                using alps::numeric::operator+=;
                using alps::numeric::operator*;
                using alps::numeric::check_size;

                B::operator()(val);
                if(B::count() == (1UL << m_ac_sum2.size())) {
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

                    // in other words: (B::count() % (1L << i) == 0)
                    if (!(B::count() & ((1ll << i) - 1))) {
                        m_ac_sum2[i] += m_ac_partial[i] * m_ac_partial[i];
                        m_ac_sum[i] += m_ac_partial[i];
                        m_ac_count[i]++;
                        m_ac_partial[i] = T();
                        check_size(m_ac_partial[i], val);
                    }
                }
            }

            template<typename T, typename B>
            void Accumulator<T, binning_analysis_tag, B>::save(hdf5::archive & ar) const {
                B::save(ar);
                if (B::count())
                    ar["tau/partialbin"] = m_ac_sum;
                ar["tau/data"] = m_ac_sum2;
                ar["tau/ac_count"] = m_ac_count; // FIXME: proper dataset name? to be saved always?
                ar["tau/ac_partial"] = m_ac_partial;  // FIXME: proper dataset name? to be saved always?
            }

            template<typename T, typename B>
            void Accumulator<T, binning_analysis_tag, B>::load(hdf5::archive & ar) { // TODO: make archive const
                B::load(ar);
                if (ar.is_data("tau/partialbin"))
                    ar["tau/partialbin"] >> m_ac_sum;
                ar["tau/data"] >> m_ac_sum2;
                if (ar.is_data("tau/ac_count"))
                    ar["tau/ac_count"] >> m_ac_count; // FIXME: proper dataset name?
                if (ar.is_data("tau/ac_partial"))
                    ar["tau/ac_partial"] >> m_ac_partial;  // FIXME: proper dataset name?
            }

            template<typename T, typename B>
            bool Accumulator<T, binning_analysis_tag, B>::can_load(hdf5::archive & ar) { // TODO: make archive const
                using alps::hdf5::get_extent;
                const char name[]="tau/data";
                const std::size_t ndim=get_extent(T()).size()+1;
                return B::can_load(ar) &&
                        detail::archive_trait<T>::can_load(ar, name, ndim); // FIXME: `T` should rather be `error_type`, defined at class level
            }

            template<typename T, typename B>
            void Accumulator<T, binning_analysis_tag, B>::reset() {
                B::reset();
                m_ac_sum = std::vector<T>();
                m_ac_sum2 = std::vector<T>();
                m_ac_partial = std::vector<T>();
                m_ac_count = std::vector<typename count_type<B>::type>();
            }

#ifdef ALPS_HAVE_MPI
            template<typename T, typename B>
            void Accumulator<T, binning_analysis_tag, B>::collective_merge(
                  alps::mpi::communicator const & comm
                , int root
            ) {

                if (comm.rank() == root) {
                    B::collective_merge(comm, root);
                    typedef typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type mean_scalar_type;
                    std::size_t size = alps::mpi::all_reduce(comm, m_ac_count.size(), alps::mpi::maximum<std::size_t>());

                    m_ac_count.resize(size);
                    B::reduce_if(comm, std::vector<typename count_type<B>::type>(m_ac_count), m_ac_count, std::plus<typename count_type<B>::type>(), root);

                    m_ac_sum.resize(size);
                    alps::numeric::rectangularize(m_ac_sum);
                    B::reduce_if(comm, std::vector<T>(m_ac_sum), m_ac_sum, std::plus<mean_scalar_type>(), root);

                    m_ac_sum2.resize(size);
                    alps::numeric::rectangularize(m_ac_sum2);
                    B::reduce_if(comm, std::vector<T>(m_ac_sum2), m_ac_sum2, std::plus<mean_scalar_type>(), root);

                } else
                    const_cast<Accumulator<T, binning_analysis_tag, B> const *>(this)->collective_merge(comm, root);
            }

            template<typename T, typename B>
            void Accumulator<T, binning_analysis_tag, B>::collective_merge(
                  alps::mpi::communicator const & comm
                , int root
            ) const {
                B::collective_merge(comm, root);
                if (comm.rank() == root)
                    throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
                else {
                    typedef typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type mean_scalar_type;

                    std::size_t size = alps::mpi::all_reduce(comm, m_ac_count.size(), alps::mpi::maximum<std::size_t>());
                    {
                        std::vector<typename count_type<B>::type> count(m_ac_count);
                        count.resize(size);
                        B::reduce_if(comm, count, std::plus<typename count_type<B>::type>(), root);
                    }
                    {
                        std::vector<T> sum(m_ac_sum);
                        sum.resize(size);
                        alps::numeric::rectangularize(sum);
                        B::reduce_if(comm, sum, std::plus<mean_scalar_type>(), root);
                    }
                    {
                        std::vector<T> sum2(m_ac_sum2);
                        sum2.resize(size);
                        alps::numeric::rectangularize(sum2);
                        B::reduce_if(comm, sum2, std::plus<mean_scalar_type>(), root);
                    }
                }
            }
#endif

            #define ALPS_ACCUMULATOR_INST_BINNING_ANALYSIS_ACC(r, data, T)                         \
                template class Accumulator<T, binning_analysis_tag,                                \
                                           Accumulator<T, error_tag,                               \
                                           Accumulator<T, mean_tag,                                \
                                           Accumulator<T, count_tag,                               \
                                           AccumulatorBase<T>>>>>;

            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_BINNING_ANALYSIS_ACC, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)

            //
            // Result<T, binning_analysis_tag, B>
            //

            template<typename T, typename B>
            Result<T, binning_analysis_tag, B>::Result()
                : B()
                , m_ac_autocorrelation()
                , m_ac_errors()
            {}

            template<typename T, typename B>
            auto Result<T, binning_analysis_tag, B>::error(std::size_t bin_level) const -> error_type const {
                if (m_ac_errors.size() < 2)
                    return alps::numeric::inf<error_type>(B::error()); // FIXME: we call error() only to know the data size
                // AG: Seems to be wrong? (see [https://github.com/ALPSCore/ALPSCore/issues/184])
                // return m_ac_errors[bin_level >= m_ac_errors.size() ? 0 : bin_level];
                return m_ac_errors[std::min(m_ac_errors.size()-1, bin_level)];
            }

            template<typename T, typename B>
            void Result<T, binning_analysis_tag, B>::save(hdf5::archive & ar) const {
                B::save(ar);
                ar["error_bins"]=m_ac_errors;
                ar["tau"] = m_ac_autocorrelation;
            }

            template<typename T, typename B>
            void Result<T, binning_analysis_tag, B>::load(hdf5::archive & ar) {
                B::load(ar);
                ar["error_bins"] >> m_ac_errors;
                ar["tau"] >> m_ac_autocorrelation;
            }

            template<typename T, typename B>
            bool Result<T, binning_analysis_tag, B>::can_load(hdf5::archive & ar) { // TODO: make archive const
                using alps::hdf5::get_extent;
                const char name[]="tau";
                const std::size_t ndim=get_extent(T()).size();
                return B::can_load(ar) &&
                        detail::archive_trait<T>::can_load(ar, name, ndim); // FIXME: `T` should rather be `error_type`, defined at class level
            }

            template<typename T, typename B>
            void Result<T, binning_analysis_tag, B>::negate() {
                // using alps::numeric::operator-;
                // for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                //     *it = -*it;
                B::negate();
            }

            template<typename T, typename B>
            void Result<T, binning_analysis_tag, B>::inverse() {
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

            #define NUMERIC_FUNCTION_DECL(FUNCTION_NAME)                    \
                template<typename T, typename B>                            \
                void Result<T, binning_analysis_tag, B>:: FUNCTION_NAME()

            NUMERIC_FUNCTION_DECL(sin) {
                B::sin();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs(cos(this->mean()) * *it);
            }

            NUMERIC_FUNCTION_DECL(cos) {
                B::cos();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs(-sin(this->mean()) * *it);
            }

            NUMERIC_FUNCTION_DECL(tan) {
                B::tan();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs(error_scalar_type(1) / (cos(this->mean()) * cos(this->mean())) * *it);
            }

            NUMERIC_FUNCTION_DECL(sinh) {
                B::sinh();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs(cosh(this->mean()) * *it);
            }

            NUMERIC_FUNCTION_DECL(cosh) {
                B::cosh();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs(sinh(this->mean()) * *it);
            }

            NUMERIC_FUNCTION_DECL(tanh) {
                B::tanh();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs(error_scalar_type(1) / (cosh(this->mean()) * cosh(this->mean())) * *it);
            }

            NUMERIC_FUNCTION_DECL(asin) {
                B::asin();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs(error_scalar_type(1) / sqrt( -this->mean() * this->mean() + error_scalar_type(1) ) * *it);
            }

            NUMERIC_FUNCTION_DECL(acos) {
                B::acos();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs(error_scalar_type(-1) / sqrt( -this->mean() * this->mean() + error_scalar_type(1) ) * *it);
            }

            NUMERIC_FUNCTION_DECL(atan) {
                B::atan();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs(error_scalar_type(1) / (this->mean() * this->mean() + error_scalar_type(1)) * *it);
            }

            // abs does not change the error, so nothing has to be done ...

            NUMERIC_FUNCTION_DECL(sq) {
                B::sq();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs( this->mean() * (*it) * error_scalar_type(2) );
            }

            NUMERIC_FUNCTION_DECL(sqrt) {
                B::sqrt();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs(*it / ( sqrt(this->mean()) * error_scalar_type(2) ));
            }

            NUMERIC_FUNCTION_DECL(cb) {
                B::cb();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs( sq(this->mean()) * (*it) * error_scalar_type(3) );
            }

            NUMERIC_FUNCTION_DECL(cbrt) {
                B::cbrt();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs(*it / ( sq(cbrt(this->mean())) * error_scalar_type(3) ));
            }

            NUMERIC_FUNCTION_DECL(exp) {
                B::exp();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = exp(this->mean()) * *it;
            }

            NUMERIC_FUNCTION_DECL(log) {
                B::log();
                NUMERIC_FUNCTION_USING
                for (error_iterator it = m_ac_errors.begin(); it != m_ac_errors.end(); ++it)
                    *it = abs(*it / this->mean());
            }

            #define ALPS_ACCUMULATOR_INST_BINNING_ANALYSIS_RESULT(r, data, T)                         \
                template class Result<T, binning_analysis_tag,                                \
                                           Result<T, error_tag,                               \
                                           Result<T, mean_tag,                                \
                                           Result<T, count_tag,                               \
                                           ResultBase<T>>>>>;

            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_BINNING_ANALYSIS_RESULT, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
        }
    }
}
