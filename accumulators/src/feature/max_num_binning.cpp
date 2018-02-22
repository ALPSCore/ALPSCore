/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/accumulators/feature/max_num_binning.hpp>
#include <alps/accumulators/feature/binning_analysis.hpp>
#include <alps/hdf5/vector.hpp>

#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#define ALPS_ACCUMULATOR_VALUE_TYPES_SEQ BOOST_PP_TUPLE_TO_SEQ(ALPS_ACCUMULATOR_VALUE_TYPES_SIZE, (ALPS_ACCUMULATOR_VALUE_TYPES))

namespace alps {
    namespace accumulators {
        namespace impl {

            //
            // Accumulator<T, max_num_binning_tag, B>
            //

            template<typename T, typename B>
            Accumulator<T, max_num_binning_tag, B>::Accumulator()
                : B()
                , m_mn_max_number(128)
                , m_mn_elements_in_bin(0)
                , m_mn_elements_in_partial(0)
                , m_mn_partial(T())
            {}

            template<typename T, typename B>
            Accumulator<T, max_num_binning_tag, B>::Accumulator(Accumulator const & arg)
                : B(arg)
                , m_mn_max_number(arg.m_mn_max_number)
                , m_mn_elements_in_bin(arg.m_mn_elements_in_bin)
                , m_mn_elements_in_partial(arg.m_mn_elements_in_partial)
                , m_mn_partial(arg.m_mn_partial)
                , m_mn_bins(arg.m_mn_bins)
            {}

            template<typename T, typename B>
            void Accumulator<T, max_num_binning_tag, B>::operator()(T const & val) {
                using alps::numeric::operator+=;
                using alps::numeric::operator+;
                using alps::numeric::operator/;
                using alps::numeric::check_size;

                B::operator()(val);

                if (!m_mn_elements_in_bin) {
                    m_mn_bins.push_back(val);
                    m_mn_elements_in_bin = 1;
                } else {
                    check_size(m_mn_bins[0], val);
                    check_size(m_mn_partial, val);
                    m_mn_partial += val;
                    ++m_mn_elements_in_partial;
                }

                // TODO: make library for scalar type
                typename alps::numeric::scalar<T>::type elements_in_bin = m_mn_elements_in_bin;
                typename alps::numeric::scalar<typename mean_type<B>::type>::type two = 2;

                if (m_mn_elements_in_partial == m_mn_elements_in_bin && m_mn_bins.size() >= m_mn_max_number) {
                    if (m_mn_max_number % 2 == 1) {
                        m_mn_partial += m_mn_bins[m_mn_max_number - 1];
                        m_mn_elements_in_partial += m_mn_elements_in_bin;
                    }
                    for (typename count_type<T>::type i = 0; i < m_mn_max_number / 2; ++i)
                        m_mn_bins[i] = (m_mn_bins[2 * i] + m_mn_bins[2 * i + 1]) / two;
                    m_mn_bins.erase(m_mn_bins.begin() + m_mn_max_number / 2, m_mn_bins.end());
                    m_mn_elements_in_bin *= (typename count_type<T>::type)2;
                }
                if (m_mn_elements_in_partial == m_mn_elements_in_bin) {
                    m_mn_bins.push_back(m_mn_partial / elements_in_bin);
                    m_mn_partial = T();
                    m_mn_elements_in_partial = 0;
                }
            }

            template<typename T, typename B>
            void Accumulator<T, max_num_binning_tag, B>::save(hdf5::archive & ar) const {
                B::save(ar);
                if (B::count()) {
                    ar["timeseries/partialbin"] = m_mn_partial;
                    ar["timeseries/partialbin/@count"] = m_mn_elements_in_partial;
                }
                ar["timeseries/data"] = m_mn_bins;
                ar["timeseries/data/@binningtype"] = "linear";
                ar["timeseries/data/@minbinsize"] = 0; // TODO: what should we put here?
                ar["timeseries/data/@binsize"] = m_mn_elements_in_bin;
                ar["timeseries/data/@maxbinnum"] = m_mn_max_number;
            }

            template<typename T, typename B>
            void Accumulator<T, max_num_binning_tag, B>::load(hdf5::archive & ar) { // TODO: make archive const
                B::load(ar);
                ar["timeseries/data"] >> m_mn_bins;
                ar["timeseries/data/@binsize"] >> m_mn_elements_in_bin;
                ar["timeseries/data/@maxbinnum"] >> m_mn_max_number;
                if (ar.is_data("timeseries/partialbin")) {
                    ar["timeseries/partialbin"] >> m_mn_partial;
                    ar["timeseries/partialbin/@count"] >> m_mn_elements_in_partial;
                }
            }

            template<typename T, typename B>
            bool Accumulator<T, max_num_binning_tag, B>::can_load(hdf5::archive & ar) { // TODO: make archive const
                using alps::hdf5::get_extent;
                const char name[]="timeseries/data";
                const std::size_t ndim=get_extent(T()).size()+1;
                return B::can_load(ar) &&
                        detail::archive_trait<T>::can_load(ar, name, ndim) && // FIXME: `T` should rather be `error_type`, defined at class level
                        ar.is_attribute("timeseries/data/@binsize") &&
                        ar.is_attribute("timeseries/data/@maxbinnum");

                    // && ar.is_data(name)
                    // && ar.is_datatype<typename alps::hdf5::scalar_type<T>::type>(name)
                    // && ar.is_attribute("timeseries/data/@binsize")
                    // && ar.is_attribute("timeseries/data/@maxbinnum")
                    // && get_extent(T()).size() + 1 == ar.dimensions(name)
                    // ;
            }

            template<typename T, typename B>
            void Accumulator<T, max_num_binning_tag, B>::reset() {
                B::reset();
                m_mn_elements_in_bin = typename B::count_type();
                m_mn_elements_in_partial = typename B::count_type();
                m_mn_partial = T();
                m_mn_bins = std::vector<typename mean_type<B>::type>();
            }

#ifdef ALPS_HAVE_MPI
            template<typename T, typename B>
            void Accumulator<T, max_num_binning_tag, B>::collective_merge(alps::mpi::communicator const & comm,
                                                                          int root)
            {
                if (comm.rank() == root) {
                    B::collective_merge(comm, root);
                    if (!m_mn_bins.empty()) {
                        std::vector<typename mean_type<B>::type> local_bins(m_mn_bins), merged_bins;
                        partition_bins(comm, local_bins, merged_bins, root);
                        B::reduce_if(comm,
                                      merged_bins,
                                      m_mn_bins,
                                      std::plus<typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type>(),
                                      root);
                    }
                } else
                    const_cast<Accumulator<T, max_num_binning_tag, B> const *>(this)->collective_merge(comm, root);
            }

            template<typename T, typename B>
            void Accumulator<T, max_num_binning_tag, B>::collective_merge(alps::mpi::communicator const & comm,
                                                                          int root) const
            {
                B::collective_merge(comm, root);
                if (comm.rank() == root)
                    throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
                else if (!m_mn_bins.empty()) {
                    std::vector<typename mean_type<B>::type> local_bins(m_mn_bins), merged_bins;
                    partition_bins(comm, local_bins, merged_bins, root);
                    B::reduce_if(comm, merged_bins, std::plus<typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type>(), root);
                }
            }

            template<typename T, typename B>
            void Accumulator<T, max_num_binning_tag, B>::partition_bins(alps::mpi::communicator const & comm,
                                                                        std::vector<typename mean_type<B>::type> & local_bins,
                                                                        std::vector<typename mean_type<B>::type> & merged_bins,
                                                                        int) const
            {
                using alps::numeric::operator+;
                using alps::numeric::operator/;
                using alps::numeric::check_size;

                typename B::count_type elements_in_local_bins = alps::mpi::all_reduce(comm, m_mn_elements_in_bin, alps::mpi::maximum<typename B::count_type>());
                typename B::count_type howmany = (elements_in_local_bins - 1) / m_mn_elements_in_bin + 1;
                if (howmany > 1) {
                    typename B::count_type newbins = local_bins.size() / howmany;
                    typename alps::numeric::scalar<typename mean_type<B>::type>::type howmany_vt = howmany;
                    for (typename B::count_type i = 0; i < newbins; ++i) {
                        local_bins[i] = local_bins[howmany * i];
                        for (typename B::count_type j = 1; j < howmany; ++j)
                            local_bins[i] = local_bins[i] + local_bins[howmany * i + j];
                        local_bins[i] = local_bins[i] / howmany_vt;
                    }
                        local_bins.resize(newbins);
                }

                std::vector<std::size_t> index(comm.size());
                alps::mpi::all_gather(comm, local_bins.size(), index);
                std::size_t total_bins = std::accumulate(index.begin(), index.end(), 0);
                std::size_t perbin = total_bins < m_mn_max_number ? 1 : total_bins / m_mn_max_number;
                typename alps::numeric::scalar<typename mean_type<B>::type>::type perbin_vt = perbin;

                merged_bins.resize(perbin == 1 ? total_bins : m_mn_max_number);
                for (typename std::vector<typename mean_type<B>::type>::iterator it = merged_bins.begin(); it != merged_bins.end(); ++it)
                    check_size(*it, local_bins[0]);

                std::size_t start = std::accumulate(index.begin(), index.begin() + comm.rank(), 0);
                for (std::size_t i = start / perbin, j = start % perbin, k = 0; i < merged_bins.size() && k < local_bins.size(); ++k) {
                    merged_bins[i] = merged_bins[i] + local_bins[k] / perbin_vt;
                    if (++j == perbin)
                        ++i, j = 0;
                }
            }
#endif

            #define ALPS_ACCUMULATOR_INST_MAX_NUM_BINNING_ACC(r, data, T)                          \
                template class Accumulator<T, max_num_binning_tag,                                 \
                                           Accumulator<T, binning_analysis_tag,                    \
                                           Accumulator<T, error_tag,                               \
                                           Accumulator<T, mean_tag,                                \
                                           Accumulator<T, count_tag,                               \
                                           AccumulatorBase<T>>>>>>;

            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_MAX_NUM_BINNING_ACC, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)

            //
            // Result<T, max_num_binning_tag, B>
            //

            template<typename T, typename B>
            Result<T, max_num_binning_tag, B>::Result()
                : B()
                , m_mn_max_number(0)
                , m_mn_elements_in_bin(0)
                , m_mn_count(typename B::count_type())
                , m_mn_mean(typename mean_type<B>::type())
                , m_mn_error(typename error_type<B>::type())
                , m_mn_cannot_rebin(false)
                , m_mn_jackknife_valid(false)
                , m_mn_data_is_analyzed(true)
                , m_mn_jackknife_bins(0)
            {}

            template<typename T, typename B>
            typename B::count_type Result<T, max_num_binning_tag, B>::count() const {
                if (!m_mn_data_is_analyzed) {
                    return m_mn_elements_in_bin * m_mn_bins.size();
                }
                else {
                    return m_mn_count;
                };
                //analyze();
            }

            template<typename T, typename B>
            typename mean_type<B>::type const & Result<T, max_num_binning_tag, B>::mean() const {
                analyze();
                return m_mn_mean;
            }

            template<typename T, typename B>
            typename error_type<B>::type const & Result<T, max_num_binning_tag, B>::error() const {
                analyze();
                return m_mn_error;
            }

            // Seem to be broken ...
            /*
            template<typename T, typename B> template <typename A>
            typename std::enable_if<has_feature<A, max_num_binning_tag>::value,
                                      typename covariance_type<B>::type
                                      >::type Result<T, max_num_binning_tag, B>::covariance(A const & obs) const
            {
                using alps::numeric::operator+;
                using alps::numeric::operator/;
                using alps::numeric::outer_product;

                generate_jackknife();
                obs.generate_jackknife();
                if (m_mn_jackknife_bins.size() != obs.m_mn_jackknife_bins.size())
                    throw std::runtime_error("Unequal number of bins in calculation of covariance matrix" + ALPS_STACKTRACE);
                if (!m_mn_jackknife_bins.size() || !obs.m_mn_jackknife_bins.size())
                    throw std::runtime_error("No binning information available for calculation of covariances" + ALPS_STACKTRACE);

                typename alps::numeric::scalar<typename mean_type<B>::type>::type bin_number = m_mn_bins.size();

                typename mean_type<B>::type unbiased_mean_1;
                for (typename std::vector<typename mean_type<B>::type>::const_iterator it = m_mn_jackknife_bins.begin() + 1; it != m_mn_jackknife_bins.end(); ++it)
                    unbiased_mean_1 = unbiased_mean_1 + *it / bin_number;

                typename mean_type<B>::type unbiased_mean_2;
                for (typename std::vector<typename mean_type<B>::type>::const_iterator it = obs.m_mn_jackknife_bins.begin() + 1; it != obs.m_mn_jackknife_bins.end(); ++it)
                    unbiased_mean_2 = unbiased_mean_2 + *it / bin_number;

                typename covariance_type<B>::type cov = outer_product(m_mn_jackknife_bins[1], obs.m_mn_jackknife_bins[1]);
                for (typename B::count_type i = 1; i < m_mn_bins.size(); ++i)
                    cov += outer_product(m_mn_jackknife_bins[i + 1], obs.m_mn_jackknife_bins[i + 1]);
                cov /= bin_number;
                cov -= outer_product(unbiased_mean_1, unbiased_mean_2);
                cov *= bin_number - 1;
                return cov;
            }

            // Adapted from http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Covariance
            // It is a two-pass algorithm, which first calculates estimates for the mean and then performs
            // the stable algorithm on the residuals. According to literature and local authorities, this
            // is the most accurate and stable way to calculate variances.
            template<typename T, typename B> template <typename A> typename std::enable_if<
                has_feature<A, max_num_binning_tag>::value, typename covariance_type<B>::type
                >::type Result<T, max_num_binning_tag, B>::accurate_covariance(A const & obs) const {
                using alps::numeric::operator+;
                using alps::numeric::operator-;
                using alps::numeric::operator/;
                using alps::numeric::outer_product;

                generate_jackknife();
                obs.generate_jackknife();
                if (m_mn_jackknife_bins.size() != obs.m_mn_jackknife_bins.size())
                    throw std::runtime_error("Unequal number of bins in calculation of covariance matrix" + ALPS_STACKTRACE);
                if (!m_mn_jackknife_bins.size() || !obs.m_mn_jackknife_bins.size())
                    throw std::runtime_error("No binning information available for calculation of covariances" + ALPS_STACKTRACE);

                typedef typename alps::numeric::scalar<typename mean_type<B>::type>::type scalar_type;
                scalar_type bin_number = m_mn_bins.size();

                typename mean_type<B>::type unbiased_mean_1;
                for (typename std::vector<typename mean_type<B>::type>::const_iterator it = m_mn_jackknife_bins.begin() + 1; it != m_mn_jackknife_bins.end(); ++it)
                    unbiased_mean_1 = unbiased_mean_1 + *it / bin_number;

                typename mean_type<B>::type unbiased_mean_2;
                for (typename std::vector<typename mean_type<B>::type>::const_iterator it = obs.m_mn_jackknife_bins.begin() + 1; it != obs.m_mn_jackknife_bins.end(); ++it)
                    unbiased_mean_2 = unbiased_mean_2 + *it / bin_number;

                std::vector<typename mean_type<B>::type> X(m_mn_bins.size());
                std::vector<typename mean_type<B>::type> Y(m_mn_bins.size());
                for (typename B::count_type i = 0; i < m_mn_bins.size(); ++i) {
                    X[i] = m_mn_jackknife_bins[i + 1] - unbiased_mean_1;
                    Y[i] = obs.m_mn_jackknife_bins[i + 1] - unbiased_mean_2;
                }

                typename mean_type<B>::type xbar;
                typename mean_type<B>::type ybar;
                typename covariance_type<B>::type cov = outer_product(xbar, ybar);
                for (typename B::count_type i = 0; i < m_mn_bins.size(); ++i) {
                    typename mean_type<B>::type delta_x = X[i] - xbar;
                    typename mean_type<B>::type delta_y = Y[i] - ybar;
                    xbar = xbar + delta_x / scalar_type(i + 1);
                    cov += outer_product(X[i] - xbar, delta_y);
                    ybar = ybar + delta_y / scalar_type(i + 1);
                }
                cov /= bin_number;
                cov *= bin_number - 1;
                return cov;
            }*/

            template<typename T, typename B>
            void Result<T, max_num_binning_tag, B>::save(hdf5::archive & ar) const {
                B::save(ar);
                analyze();
                ar["count"] = m_mn_count;
                ar["@cannotrebin"] = m_mn_cannot_rebin;
                ar["mean/value"] = m_mn_mean;
                ar["mean/error"] = m_mn_error;
                ar["timeseries/data"] = m_mn_bins;
                ar["timeseries/data/@binsize"] = m_mn_elements_in_bin;
                ar["timeseries/data/@maxbinnum"] = m_mn_max_number;
                ar["timeseries/data/@binningtype"] = "linear";
                ar["timeseries/data/@jacknife_valid"] = m_mn_jackknife_valid;
                if (m_mn_jackknife_valid) {
                    ar["jacknife/data"] = m_mn_jackknife_bins;
                    ar["jacknife/data/@binningtype"] = "linear";
                }
            }

            template<typename T, typename B>
            void Result<T, max_num_binning_tag, B>::load(hdf5::archive & ar) {
                B::load(ar);
                ar["timeseries/data"] >> m_mn_bins;
                ar["timeseries/data/@binsize"] >> m_mn_elements_in_bin;
                ar["timeseries/data/@maxbinnum"] >> m_mn_max_number;
                ar["timeseries/data/@jacknife_valid"] >> m_mn_jackknife_valid;

                ar["count"] >> m_mn_count;
                ar["@cannotrebin"] >> m_mn_cannot_rebin;
                ar["mean/value"] >> m_mn_mean;
                ar["mean/error"] >> m_mn_error;
                if (m_mn_jackknife_valid) {
                    ar["jacknife/data"] >> m_mn_jackknife_bins;
                }
            }

            template<typename T, typename B>
            bool Result<T, max_num_binning_tag, B>::can_load(hdf5::archive & ar) { // TODO: make archive const
                using alps::hdf5::get_extent;
                const char name[]="timeseries/data";
                const std::size_t ndim=get_extent(T()).size()+1;

                return B::can_load(ar) &&
                        detail::archive_trait<T>::can_load(ar, name, ndim) && // FIXME: `T` should rather be `error_type`, defined at class level
                        ar.is_attribute("timeseries/data/@binsize") &&
                        ar.is_attribute("timeseries/data/@maxbinnum");
            }

#define NUMERIC_FUNCTION_IMPLEMENTATION(FUNCTION_NAME)                                            \
            template<typename T, typename B>                                                      \
            void Result<T, max_num_binning_tag, B>:: FUNCTION_NAME () {                           \
                using alps::numeric::sq;                                                          \
                using alps::numeric::sq;                                                          \
                using alps::numeric::cbrt;                                                        \
                using alps::numeric::cbrt;                                                        \
                using alps::numeric::cb;                                                          \
                using alps::numeric::cb;                                                          \
                using std::sqrt;                                                                  \
                using alps::numeric::sqrt;                                                        \
                using std::exp;                                                                   \
                using alps::numeric::exp;                                                         \
                using std::log;                                                                   \
                using alps::numeric::log;                                                         \
                using std::abs;                                                                   \
                using alps::numeric::abs;                                                         \
                using std::pow;                                                                   \
                using alps::numeric::pow;                                                         \
                using std::sin;                                                                   \
                using alps::numeric::sin;                                                         \
                using std::cos;                                                                   \
                using alps::numeric::cos;                                                         \
                using std::tan;                                                                   \
                using alps::numeric::tan;                                                         \
                using std::sinh;                                                                  \
                using alps::numeric::sinh;                                                        \
                using std::cosh;                                                                  \
                using alps::numeric::cosh;                                                        \
                using std::tanh;                                                                  \
                using alps::numeric::tanh;                                                        \
                using std::asin;                                                                  \
                using alps::numeric::asin;                                                        \
                using std::acos;                                                                  \
                using alps::numeric::acos;                                                        \
                using std::atan;                                                                  \
                using alps::numeric::atan;                                                        \
                typedef typename value_type<B>::type (*fptr_type)(typename value_type<B>::type);  \
                fptr_type fptr=& FUNCTION_NAME;                                                   \
                transform(fptr);                                                                  \
                B:: FUNCTION_NAME ();                                                             \
            }

            NUMERIC_FUNCTION_IMPLEMENTATION(sin)
            NUMERIC_FUNCTION_IMPLEMENTATION(cos)
            NUMERIC_FUNCTION_IMPLEMENTATION(tan)
            NUMERIC_FUNCTION_IMPLEMENTATION(sinh)
            NUMERIC_FUNCTION_IMPLEMENTATION(cosh)
            NUMERIC_FUNCTION_IMPLEMENTATION(tanh)
            NUMERIC_FUNCTION_IMPLEMENTATION(asin)
            NUMERIC_FUNCTION_IMPLEMENTATION(acos)
            NUMERIC_FUNCTION_IMPLEMENTATION(atan)
            NUMERIC_FUNCTION_IMPLEMENTATION(abs)
            NUMERIC_FUNCTION_IMPLEMENTATION(sq)
            NUMERIC_FUNCTION_IMPLEMENTATION(sqrt)
            NUMERIC_FUNCTION_IMPLEMENTATION(cb)
            NUMERIC_FUNCTION_IMPLEMENTATION(cbrt)
            NUMERIC_FUNCTION_IMPLEMENTATION(exp)
            NUMERIC_FUNCTION_IMPLEMENTATION(log)

#undef NUMERIC_FUNCTION_IMPLEMENTATION

            template<typename T, typename B>
            void Result<T, max_num_binning_tag, B>::negate()
            {
                using alps::numeric::negate;

                typedef typename value_type<B>::type value_type;
                transform(negate<value_type>());
                B::negate();
            }

            template<typename T, typename B>
            void Result<T, max_num_binning_tag, B>::inverse()
            {
                using alps::numeric::invert;

                typedef typename value_type<B>::type value_type;
                transform(invert<value_type>());
                B::inverse();
            }

            template<typename T, typename B>
            void Result<T, max_num_binning_tag, B>::generate_jackknife() const {
                using alps::numeric::operator-;
                using alps::numeric::operator+;
                using alps::numeric::operator*;
                using alps::numeric::operator/;
                typedef typename alps::numeric::scalar<typename mean_type<B>::type>::type scalar_type;
                // build jackknife data structure
                if (!m_mn_bins.empty() && !m_mn_jackknife_valid) {
                    if (m_mn_cannot_rebin)
                        throw std::runtime_error("Cannot build jackknife data structure after nonlinear operations" + ALPS_STACKTRACE);
                    m_mn_jackknife_bins.clear();
                    m_mn_jackknife_bins.resize(m_mn_bins.size() + 1);
                    // Order-N initialization of jackknife data structure
                    //    m_mn_jackknife_bins[0]   =  <x>
                    //    m_mn_jackknife_bins[i+1] =  <x_i>_{jacknife}
                    scalar_type bin_number = m_mn_bins.size();
                    for(std::size_t j = 0; j < m_mn_bins.size(); ++j) // m_mn_jackknife_bins[0] = \sum_{j} m_mn_bins[j]
                        m_mn_jackknife_bins[0] = m_mn_jackknife_bins[0] + m_mn_bins[j];
                    for(std::size_t i = 0; i < m_mn_bins.size(); ++i) // m_mn_jackknife_bins[i+1] = \sum_{j != i} m_mn_bins[j] / #m_mn_bins
                      m_mn_jackknife_bins[i + 1] = (m_mn_jackknife_bins[0] - m_mn_bins[i]) / (bin_number - static_cast<scalar_type>(1));
                    m_mn_jackknife_bins[0] = m_mn_jackknife_bins[0] / bin_number; // m_mn_jackknife_bins[0] is the jacknife mean...
                }
                m_mn_jackknife_valid = true;
            }

            template<typename T, typename B>
            void Result<T, max_num_binning_tag, B>::analyze() const {
                using alps::numeric::sq;
                using std::sqrt;
                using alps::numeric::sqrt;
                using alps::numeric::operator-;
                using alps::numeric::operator+;
                using alps::numeric::operator*;
                using alps::numeric::operator/;
                typedef typename alps::numeric::scalar<typename mean_type<B>::type>::type scalar_type;

                if (m_mn_bins.empty())
                    throw std::runtime_error("No Measurement" + ALPS_STACKTRACE);
                if (!m_mn_data_is_analyzed) {
                    m_mn_count = m_mn_elements_in_bin * m_mn_bins.size();
                    generate_jackknife();
                    if (m_mn_jackknife_bins.size()) {
                        typename mean_type<B>::type unbiased_mean = typename mean_type<B>::type();
                        scalar_type bin_number = m_mn_bins.size();
                        for (typename std::vector<typename mean_type<B>::type>::const_iterator it = m_mn_jackknife_bins.begin() + 1;
                              it != m_mn_jackknife_bins.end(); ++it)
                            unbiased_mean = unbiased_mean + *it / bin_number;
                        m_mn_mean = m_mn_jackknife_bins[0] - (unbiased_mean - m_mn_jackknife_bins[0]) * (bin_number - static_cast<scalar_type>(1));
                        m_mn_error = typename error_type<B>::type();
                        for (std::size_t i = 0; i < m_mn_bins.size(); ++i)
                            m_mn_error = m_mn_error + sq(m_mn_jackknife_bins[i + 1] - unbiased_mean);
                        m_mn_error = sqrt(m_mn_error / bin_number * (bin_number - static_cast<scalar_type>(1)));
                    }
                }
                m_mn_data_is_analyzed = true;
            }

            template<typename T>
            using result_t = Result<T, max_num_binning_tag,
                                      Result<T, binning_analysis_tag,
                                      Result<T, error_tag,
                                      Result<T, mean_tag,
                                      Result<T, count_tag,
                                      ResultBase<T>>>>>>;
            template<typename T>
            using covariance_t = typename covariance_type<Result<T, binning_analysis_tag,
                                                          Result<T, error_tag,
                                                          Result<T, mean_tag,
                                                          Result<T, count_tag,
                                                          ResultBase<T>>>>>>::type;

            #define ALPS_ACCUMULATOR_INST_MAX_NUM_BINNING_RESULT(r, data, T)                                      \
                template class Result<T, max_num_binning_tag,                                                     \
                                      Result<T, binning_analysis_tag,                                             \
                                      Result<T, error_tag,                                                        \
                                      Result<T, mean_tag,                                                         \
                                      Result<T, count_tag,                                                        \
                                      ResultBase<T>>>>>>;                                                         \
                /*template covariance_t<T> result_t<T>::covariance<result_t<T>>(const result_t<T>&) const;          \
                template covariance_t<T> result_t<T>::accurate_covariance<result_t<T>>(const result_t<T>&) const;*/

            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_INST_MAX_NUM_BINNING_RESULT, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
        }
    }
}
