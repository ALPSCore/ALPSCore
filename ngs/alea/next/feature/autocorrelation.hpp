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

#ifndef ALPS_NGS_ACCUMULATOR_AUTOCORRELATION_HPP
#define ALPS_NGS_ACCUMULATOR_AUTOCORRELATION_HPP

#include <alps/ngs/alea/next/feature.hpp>
#include <alps/ngs/alea/next/feature/mean.hpp>
#include <alps/ngs/alea/next/feature/count.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/numeric.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/short_print.hpp>

#include <boost/mpl/if.hpp>
#include <boost/utility.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/is_integral.hpp>

#include <stdexcept>

namespace alps {
    namespace accumulator {
        // this should be called namespace tag { struct autocorrelation; }
        // but gcc <= 4.4 has lookup error, so name it different
        struct autocorrelation_tag;

        template<typename T> struct autocorrelation_type {
            typedef std::vector<typename value_type<T>::type> type;
        };

        template<typename T> struct has_feature<T, autocorrelation_tag> {
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
                  typename has_feature<A, autocorrelation_tag>::type
                , typename autocorrelation_type<A>::type
            >::type autocorrelation_impl(A const & acc) {
                return autocorrelation(acc);
            }

            template<typename A> typename boost::disable_if<
                  typename has_feature<A, autocorrelation_tag>::type
                , typename autocorrelation_type<A>::type
            >::type autocorrelation_impl(A const & acc) {
                throw std::runtime_error(std::string(typeid(A).name()) + " has no autocorrelation-method" + ALPS_STACKTRACE);
                return *static_cast<typename autocorrelation_type<A>::type *>(NULL);
            }
        }

        namespace impl {

            template<typename T, typename B> struct Accumulator<T, autocorrelation_tag, B> : public B {

                public:
                    typedef Result<T, autocorrelation_tag, typename B::result_type> result_type;

                    // TODO: implement ...
                    template<typename ArgumentPack> Accumulator(ArgumentPack const & args)
                        : B(args)
                        , m_ac_partial(0)
                        , m_ac_bins(0)
                    {}

                    Accumulator()
                        : B()
                        , m_ac_partial(0)
                        , m_ac_bins(0)
                    {}

                    Accumulator(Accumulator const & arg)
                        : B(arg)
                        , m_ac_partial(arg.m_ac_partial)
                        , m_ac_bins(arg.m_ac_bins)
                    {}

                    std::vector<T> const autocorrelation() const {
                        using alps::ngs::numeric::operator*;
                        using alps::ngs::numeric::operator-;
                        using alps::ngs::numeric::operator/;
                        using std::sqrt;
                        using alps::ngs::numeric::sqrt;
                        typename alps::hdf5::scalar_type<T>::type cnt = B::count() - 1;
                        // TODO: probably we need less than count(), since we remove the partials ...
                        std::vector<T> result = m_ac_bins - m_ac_partial * m_ac_partial;
                        for (typename std::vector<T>::iterator it = result.begin(); it != result.end(); ++it)
                            *it = sqrt(*it / cnt);
                        return result;
                    }

                    void operator()(T const & val) {
                        using alps::ngs::numeric::operator+;
                        using alps::ngs::numeric::operator+=;
                        using alps::ngs::numeric::operator-;
                        using alps::ngs::numeric::operator*;
                        using alps::ngs::numeric::operator-;
                        using alps::ngs::numeric::detail::check_size;

                        B::operator()(val);
                        
                        if(B::count() == (1 << m_ac_bins.size())) {
                            m_ac_bins.push_back(T());
                            check_size(m_ac_bins.back(), val);

                            m_ac_partial.push_back(T());
                            check_size(m_ac_partial.back(), val);
                        }
                        for (unsigned i = 0; i < m_ac_bins.size(); ++i)
                            // TODO: check if this makes sence
                            if(B::count() % (1u << i) == 0) {
                                m_ac_partial[i] = B::sum() - m_ac_partial[i];
                                m_ac_bins[i] += m_ac_partial[i] * m_ac_partial[i];
                                m_ac_partial[i] = B::sum();
                            }
                    }

                    template<typename S> void print(S & os) const {
                        B::print(os);
                        os << " Tau: " << short_print(autocorrelation());
                    }

                    void save(hdf5::archive & ar) const {
                        B::save(ar);
                        if (B::count())
                            ar["tau/partialbin"] = m_ac_partial;
                        ar["tau/data"] = m_ac_bins;
                    }

                    void load(hdf5::archive & ar) { // TODO: make archive const
                        B::load(ar);
                        if (ar.is_data("tau/partialbin"))
                            ar["tau/partialbin"] >> m_ac_partial;
                        ar["tau/data"] >> m_ac_bins;
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

// #ifdef ALPS_HAVE_MPI
//                     void collective_merge(
//                           boost::mpi::communicator const & comm
//                         , int root
//                     ) {
//                         B::collective_merge(comm, root);
//                         if (comm.rank() == root) {
//                             if (!m_mn_bins.empty()) {
//                                 std::vector<typename mean_type<B>::type> local_bins(m_mn_bins);
//                                 m_mn_elements_in_bin = partition_bins(comm, local_bins);
//                                 m_mn_bins.resize(local_bins.size());
//                                 B::reduce_if(comm, local_bins, m_mn_bins, std::plus<typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type>(), 0);
//                             }
//                         } else
//                             const_cast<Accumulator<T, autocorrelation_tag, B> const *>(this)->collective_merge(comm, root);
//                     }

//                     void collective_merge(
//                           boost::mpi::communicator const & comm
//                         , int root
//                     ) const {
//                         B::collective_merge(comm, root);
//                         if (comm.rank() == root)
//                             throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
//                         else if (!m_mn_bins.empty()) {
//                             std::vector<typename mean_type<B>::type> local_bins(m_mn_bins);
//                             partition_bins(comm, local_bins);
//                             B::reduce_if(comm, local_bins, std::plus<typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type>(), root);
//                         }
//                     }

//                 private:
//                     std::size_t partition_bins (boost::mpi::communicator const & comm, std::vector<typename mean_type<B>::type> & local_bins) const {
//                         using alps::ngs::numeric::operator+;
//                         using alps::ngs::numeric::operator/;
//                         typename B::count_type elements_in_local_bins = boost::mpi::all_reduce(comm, m_mn_elements_in_bin, boost::mpi::maximum<typename B::count_type>());
//                         typename B::count_type howmany = (elements_in_local_bins - 1) / m_mn_elements_in_bin + 1;
//                         if (howmany > 1) {
//                             typename B::count_type newbins = local_bins.size() / howmany;
//                             for (typename B::count_type i = 0; i < newbins; ++i) {
//                                 local_bins[i] = local_bins[howmany * i];
//                                 for (typename B::count_type j = 1; j < howmany; ++j)
//                                     local_bins[i] = local_bins[i] + local_bins[howmany * i + j];
//                                 local_bins[i] = local_bins[i] / (typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type)howmany;

//                             }
//                             local_bins.resize(newbins);
//                         }
//                         return elements_in_local_bins;
//                     }
// #endif

                private:

                    std::vector<T> m_ac_partial;
                    std::vector<T> m_ac_bins;
            };

            template<typename T, typename B> class Result<T, autocorrelation_tag, B> : public B {

                public:
                    typedef typename alps::accumulator::autocorrelation_type<B>::type autocorrelation_type;

                    Result()
                        : B()
                        , m_ac_bins(0)
                    {}

                    template<typename A> Result(A const & acc)
                        : B(acc)
                        , m_ac_bins(autocorrelation_impl(acc))
                    {}

                    autocorrelation_type const autocorrelation() const {
                        return m_ac_bins;
                    }

                    template<typename S> void print(S & os) const {
                        B::print(os);
                        os << " Tau: " << short_print(autocorrelation());
                    }

                    void save(hdf5::archive & ar) const {
                        B::save(ar);
                        ar["tau"] = m_ac_bins;
                    }

                    void load(hdf5::archive & ar) {
                        B::load(ar);
                        ar["tau"] >> m_ac_bins;
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
                    std::vector<T> m_ac_bins;
            };

            template<typename B> class BaseWrapper<autocorrelation_tag, B> : public B {
                public:
                    virtual bool has_autocorrelation() const = 0;
            };

            template<typename T, typename B> class ResultTypeWrapper<T, autocorrelation_tag, B> : public B {
                public:
                    virtual typename autocorrelation_type<B>::type autocorrelation() const = 0;
            };

            template<typename T, typename B> class DerivedWrapper<T, autocorrelation_tag, B> : public B {
                public:
                    DerivedWrapper(): B() {}
                    DerivedWrapper(T const & arg): B(arg) {}

                    bool has_autocorrelation() const { return has_feature<T, autocorrelation_tag>::type::value; }

                    typename autocorrelation_type<B>::type autocorrelation() const { return detail::autocorrelation_impl(this->m_data); }
            };

        }
    }
}

 #endif
