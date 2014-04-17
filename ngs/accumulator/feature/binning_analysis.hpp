/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2013 by Mario Koenz <mkoenz@ethz.ch>                       *
 *                              Lukas Gamper <gamperl@gmail.com>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef ALPS_NGS_ACCUMULATOR_BINNING_ANALYSIS_HPP
#define ALPS_NGS_ACCUMULATOR_BINNING_ANALYSIS_HPP

#include <alps/ngs/accumulator/feature.hpp>
#include <alps/ngs/accumulator/parameter.hpp>
#include <alps/ngs/accumulator/feature/mean.hpp>
#include <alps/ngs/accumulator/feature/count.hpp>

#include <alps/ngs/numeric.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/short_print.hpp>

#include <alps/alea/convergence.hpp>
#include <alps/numeric/set_negative_0.hpp>

#include <boost/mpl/if.hpp>
#include <boost/utility.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/is_integral.hpp>

#include <limits>
#include <stdexcept>

namespace alps {
    namespace accumulator {
        // this should be called namespace tag { struct binning_analysis; }
        // but gcc <= 4.4 has lookup error, so name it different
        struct binning_analysis_tag;

        template<typename T> struct autocorrelation_type
            : public boost::mpl::if_<boost::is_integral<typename value_type<T>::type>, double, typename value_type<T>::type>
        {};

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

                    error_convergence converged_errors() const {
                        error_convergence conv = MAYBE_CONVERGED;
                        // TODO: implement!
                        // typename alps::accumulator::mean_type<B>::type err = error();
                        // resize_same_as(conv,err);
                        // const unsigned int range = 4;
                        // typename slice_index<convergence_type>::type it;
                        // if (binning_depth()<range) {
                        //     for (it= slices(conv).first; it!= slices(conv).second; ++it)
                        //         slice_value(conv,it) = MAYBE_CONVERGED;
                        // }
                        // else {
                        //     for (it= slices(conv).first; it!= slices(conv).second; ++it)
                        //         slice_value(conv,it) = CONVERGED;

                        //     for (unsigned int i=binning_depth()-range;i<binning_depth()-1;++i) {
                        //         result_type this_err(error(i));
                        //         for (it= slices(conv).first; it!= slices(conv).second; ++it)
                        //             if (std::abs(slice_value(this_err,it)) >= std::abs(slice_value(err,it)))
                        //                 slice_value(conv,it)=CONVERGED;
                        //             else if (std::abs(slice_value(this_err,it)) < 0.824 * std::abs(slice_value(err,it)))
                        //                 slice_value(conv,it)=NOT_CONVERGED;
                        //             else if (std::abs(slice_value(this_err,it)) <0.9* std::abs(slice_value(err,it))  &&
                        //                 slice_value(conv,it)!=NOT_CONVERGED)
                        //             slice_value(conv,it)=MAYBE_CONVERGED;
                        //     }
                        // }
                        // return conv;
                    }

                    typename alps::accumulator::error_type<B>::type const error(std::size_t bin_level = std::numeric_limits<std::size_t>::max()) const {
                        using alps::ngs::numeric::operator*;
                        using alps::ngs::numeric::operator-;
                        using alps::ngs::numeric::operator/;
                        using std::sqrt;
                        using alps::ngs::numeric::sqrt;

                        if (bin_level > m_ac_sum2.size() - 8)
                            bin_level = m_ac_sum2.size() < 8 ? 0 : m_ac_sum2.size() - 8;

                        typedef typename alps::accumulator::error_type<B>::type error_type;
                        typedef typename alps::hdf5::scalar_type<error_type>::type error_scalar_type;

                        // TODO: if not enoght bins are available, return infinity
                        if (m_ac_sum2.size() < 2)
                            return alps::ngs::numeric::inf<error_type>();

                        // TODO: make library for scalar type
                        error_scalar_type one = 1;

                        error_scalar_type binlen = 1ll << bin_level;
                        error_scalar_type N_0 = m_ac_count[0];
                        error_scalar_type N_i = m_ac_count[bin_level];
                        error_type sum_i = m_ac_sum[bin_level];
                        error_type sum2_i = m_ac_sum2[bin_level];
                        error_type var_i = (sum2_i / binlen - sum_i * sum_i / (N_i * binlen)) / (N_i * binlen);
                        return sqrt(var_i / (N_i - one));
                    }

                    // TODO: remove since 
                    typename autocorrelation_type<B>::type const autocorrelation() const {
                        using alps::ngs::numeric::operator*;
                        using alps::ngs::numeric::operator-;
                        using alps::ngs::numeric::operator/;

                        typedef typename mean_type<B>::type mean_type;

                        // TODO: make library for scalar type
                        typedef typename alps::hdf5::scalar_type<mean_type>::type mean_scalar_type;

                        // TODO: if not enoght bins are available, return infinity
                        if (m_ac_sum2.size() < 2)
                            return alps::ngs::numeric::inf<mean_type>();

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

                    void operator()(T const & val) {
                        using alps::ngs::numeric::operator+=;
                        using alps::ngs::numeric::operator*;
                        using alps::ngs::numeric::detail::check_size;

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
                        for (std::size_t i = 0; i < (m_ac_sum2.size() < 8 ? 1 : m_ac_sum2.size() - 7); ++i)
                            os << std::endl
                                << "    bin #" << std::setw(3) <<  i + 1
                                << " : " << std::setw(8) << m_ac_count[i]
                                << " entries: error = " << short_print(error(i));
                    }

                    void save(hdf5::archive & ar) const {
                        B::save(ar);
                        if (B::count())
                            ar["tau/partialbin"] = m_ac_sum;
                        ar["tau/data"] = m_ac_sum2;
                    }

                    void load(hdf5::archive & ar) { // TODO: make archive const
                        B::load(ar);
                        if (ar.is_data("tau/partialbin"))
                            ar["tau/partialbin"] >> m_ac_sum;
                        ar["tau/data"] >> m_ac_sum2;
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
                        // TODO: implement!
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
                            B::reduce_if(comm, std::vector<typename count_type<B>::type>(m_ac_count), m_ac_count, std::plus<mean_scalar_type>(), root);

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
                                B::reduce_if(comm, count, std::plus<mean_scalar_type>(), root);
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

            template<typename T, typename B> class Result<T, binning_analysis_tag, B> : public B {

                public:
                    typedef typename alps::accumulator::autocorrelation_type<B>::type autocorrelation_type;

                    Result()
                        : B()
                        , m_ac_autocorrelation()
                    {}

                    template<typename A> Result(A const & acc)
                        : B(acc)
                        , m_ac_autocorrelation(autocorrelation_impl(acc))
                    {}

                    autocorrelation_type const autocorrelation() const {
                        return m_ac_autocorrelation;
                    }

                    template<typename S> void print(S & os) const {
                        B::print(os);
                        os << " Tau: " << short_print(autocorrelation());
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

                    // TODO: add functions and operators

                private:
                    typename mean_type<B>::type m_ac_autocorrelation;
            };

            template<typename B> class BaseWrapper<binning_analysis_tag, B> : public B {
                public:
                    virtual bool has_autocorrelation() const = 0;
            };

            template<typename T, typename B> class ResultTypeWrapper<T, binning_analysis_tag, B> : public B {
                public:
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
