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

#ifndef ALPS_NGS_ACCUMULATOR_MAX_NUM_BINNING_HPP
#define ALPS_NGS_ACCUMULATOR_MAX_NUM_BINNING_HPP

#include <alps/ngs/accumulator/feature.hpp>
#include <alps/ngs/accumulator/parameter.hpp>
#include <alps/ngs/accumulator/feature/mean.hpp>
#include <alps/ngs/accumulator/feature/count.hpp>
#include <alps/ngs/accumulator/feature/error.hpp>

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
        // this should be called namespace tag { struct max_num_binning; }
        // but gcc <= 4.4 has lookup error, so name it different
        struct max_num_binning_tag;

        namespace detail {
            template<typename C, typename M> class max_num_binning_proxy {
                typedef typename std::size_t size_type;

                public:
                    max_num_binning_proxy(std::vector<M> const & bins, C const & num_elements, size_type const & max_number)
                        : m_max_number(max_number), m_num_elements(num_elements), m_bins(bins)
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

                private:

                    size_type m_max_number;
                    C m_num_elements;
                    std::vector<M> const & m_bins;
            };

            template<typename C, typename M> inline std::ostream & operator<<(std::ostream & os, max_num_binning_proxy<C, M> const & arg) {
                if (arg.bins().empty())
                    os << "No Bins";
                else
                    os << short_print(arg.bins(), 4) << "#" << arg.num_elements();
                return os;
            };
        }

        template<typename T> struct max_num_binning_type {
            typedef detail::max_num_binning_proxy<typename count_type<T>::type, typename mean_type<T>::type> type;
        };

        template<typename T> struct has_feature<T, max_num_binning_tag> {
            template<typename R, typename C> static char helper(R(C::*)() const);
            template<typename C> static char check(boost::integral_constant<std::size_t, sizeof(helper(&C::max_num_binning))>*);
            template<typename C> static double check(...);
            typedef boost::integral_constant<bool, sizeof(char) == sizeof(check<T>(0))> type;
        };

        template<typename T> typename max_num_binning_type<T>::type max_num_binning(T const & arg) {
            return arg.max_num_binning();
        }

        namespace detail {

            template<typename A> typename boost::enable_if<
                  typename has_feature<A, max_num_binning_tag>::type
                , typename max_num_binning_type<A>::type
            >::type max_num_binning_impl(A const & acc) {
                return max_num_binning(acc);
            }

            template<typename A> typename boost::disable_if<
                  typename has_feature<A, max_num_binning_tag>::type
                , typename max_num_binning_type<A>::type
            >::type max_num_binning_impl(A const & acc) {
                throw std::runtime_error(std::string(typeid(A).name()) + " has no max_num_binning-method" + ALPS_STACKTRACE);
                return *static_cast<typename max_num_binning_type<A>::type *>(NULL);
            }
        }

        namespace impl {

            template<typename T, typename B> struct Accumulator<T, max_num_binning_tag, B> : public B {

                public:
                    typedef typename alps::accumulator::max_num_binning_type<B>::type max_num_binning_type;
                    typedef Result<T, max_num_binning_tag, typename B::result_type> result_type;

                    Accumulator()
                        : B()
                        , m_mn_max_number(128)
                        , m_mn_elements_in_bin(0)
                        , m_mn_elements_in_partial(0)
                        , m_mn_partial(T())
                    {}

                    Accumulator(Accumulator const & arg)
                        : B(arg)
                        , m_mn_max_number(arg.m_mn_max_number)
                        , m_mn_elements_in_bin(arg.m_mn_elements_in_bin)
                        , m_mn_elements_in_partial(arg.m_mn_elements_in_partial)
                        , m_mn_partial(arg.m_mn_partial)
                        , m_mn_bins(arg.m_mn_bins)
                    {}

                    template<typename ArgumentPack> Accumulator(ArgumentPack const & args, typename boost::disable_if<is_accumulator<ArgumentPack>, int>::type = 0)
                        : B(args)
                        , m_mn_max_number(args[max_bin_number | 128])
                        , m_mn_elements_in_bin(0)
                        , m_mn_elements_in_partial(0)
                        , m_mn_partial(T())
                    {}

                    max_num_binning_type const max_num_binning() const {
                        return max_num_binning_type(m_mn_bins, m_mn_elements_in_bin, m_mn_max_number);
                    }

                    void operator()(T const & val) {
                        using alps::ngs::numeric::operator+=;
                        using alps::ngs::numeric::operator+;
                        using alps::ngs::numeric::operator/;
                        using alps::ngs::numeric::detail::check_size;

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
                        typename alps::hdf5::scalar_type<T>::type elements_in_bin = m_mn_elements_in_bin;
                        typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type two = 2;

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

                    template<typename S> void print(S & os) const {
                        B::print(os);
                        os << " Bins: " << max_num_binning();
                    }

                    void save(hdf5::archive & ar) const {
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

                    void load(hdf5::archive & ar) { // TODO: make archive const
                        B::load(ar);
                        ar["timeseries/data"] >> m_mn_bins;
                        ar["timeseries/data/@binsize"] >> m_mn_elements_in_bin;
                        ar["timeseries/data/@maxbinnum"] >> m_mn_max_number;
                        if (ar.is_data("timeseries/partialbin")) {
                            ar["timeseries/partialbin"] >> m_mn_partial;
                            ar["timeseries/partialbin/@count"] >> m_mn_elements_in_partial;
                        }
                    }

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::hdf5::get_extent;

                        return B::can_load(ar)
                            && ar.is_data("timeseries/data")
                            && ar.is_attribute("timeseries/data/@binsize")
                            && ar.is_attribute("timeseries/data/@maxbinnum")
                            && get_extent(T()).size() + 1 == ar.dimensions("timeseries/data")
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
                            if (!m_mn_bins.empty()) {
                                std::vector<typename mean_type<B>::type> local_bins(m_mn_bins), merged_bins;
                                partition_bins(comm, local_bins, merged_bins, root);
                                B::reduce_if(comm, merged_bins, m_mn_bins, std::plus<typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type>(), root);
                            }
                        } else
                            const_cast<Accumulator<T, max_num_binning_tag, B> const *>(this)->collective_merge(comm, root);
                    }

                    void collective_merge(
                          boost::mpi::communicator const & comm
                        , int root
                    ) const {
                        B::collective_merge(comm, root);
                        if (comm.rank() == root)
                            throw std::runtime_error("A const object cannot be root" + ALPS_STACKTRACE);
                        else if (!m_mn_bins.empty()) {
                            std::vector<typename mean_type<B>::type> local_bins(m_mn_bins), merged_bins;
                            partition_bins(comm, local_bins, merged_bins, root);
                            B::reduce_if(comm, merged_bins, std::plus<typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type>(), root);
                        }
                    }

                private:
                    void partition_bins (
                          boost::mpi::communicator const & comm
                        , std::vector<typename mean_type<B>::type> & local_bins
                        , std::vector<typename mean_type<B>::type> & merged_bins
                        , int root
                    ) const {
                        using alps::ngs::numeric::operator+;
                        using alps::ngs::numeric::operator/;
                        using alps::ngs::numeric::detail::check_size;

                        typename B::count_type elements_in_local_bins = boost::mpi::all_reduce(comm, m_mn_elements_in_bin, boost::mpi::maximum<typename B::count_type>());
                        typename B::count_type howmany = (elements_in_local_bins - 1) / m_mn_elements_in_bin + 1;
                        if (howmany > 1) {
                            typename B::count_type newbins = local_bins.size() / howmany;
                            typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type howmany_vt = howmany;
                            for (typename B::count_type i = 0; i < newbins; ++i) {
                                local_bins[i] = local_bins[howmany * i];
                                for (typename B::count_type j = 1; j < howmany; ++j)
                                    local_bins[i] = local_bins[i] + local_bins[howmany * i + j];
                                local_bins[i] = local_bins[i] / howmany_vt;
                            }
                            local_bins.resize(newbins);
                        }

                        std::vector<std::size_t> index(comm.size());
                        boost::mpi::all_gather(comm, local_bins.size(), index);
                        std::size_t total_bins = std::accumulate(index.begin(), index.end(), 0);
                        std::size_t perbin = total_bins < m_mn_max_number ? 1 : total_bins / m_mn_max_number;
                        typename alps::hdf5::scalar_type<typename mean_type<B>::type>::type perbin_vt = perbin;

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

                private:

                    std::size_t m_mn_max_number;
                    typename B::count_type m_mn_elements_in_bin, m_mn_elements_in_partial;
                    T m_mn_partial;
                    std::vector<typename mean_type<B>::type> m_mn_bins;
            };

            template<typename T, typename B> class Result<T, max_num_binning_tag, B> : public B {

                public:
                    typedef typename alps::accumulator::max_num_binning_type<B>::type max_num_binning_type;

                    Result()
                        : B()
                        , m_mn_max_number(0) 
                        , m_mn_elements_in_bin(0)
                    {}

                    template<typename A> Result(A const & acc)
                        : B(acc)
                        , m_mn_max_number(detail::max_num_binning_impl(acc).max_number())
                        , m_mn_elements_in_bin(detail::max_num_binning_impl(acc).num_elements())
                        , m_mn_bins(detail::max_num_binning_impl(acc).bins())
                    {}

                    max_num_binning_type const max_num_binning() const {
                        return max_num_binning_type(m_mn_bins, m_mn_elements_in_bin, m_mn_max_number); 
                    }

                    template<typename S> void print(S & os) const {
                        B::print(os);
                        os << " Bins: " << max_num_binning();
                    }

                    void save(hdf5::archive & ar) const {
                        B::save(ar);
                        ar["timeseries/data"] = m_mn_bins;
                        ar["timeseries/data/@binningtype"] = "linear";
                        ar["timeseries/data/@minbinsize"] = 0; // TODO: what should we put here?
                        ar["timeseries/data/@binsize"] = m_mn_elements_in_bin;
                        ar["timeseries/data/@maxbinnum"] = m_mn_max_number;
                    }

                    void load(hdf5::archive & ar) {
                        B::load(ar);
                        ar["timeseries/data"] >> m_mn_bins;
                        ar["timeseries/data/@binsize"] >> m_mn_elements_in_bin;
                        ar["timeseries/data/@maxbinnum"] >> m_mn_max_number;
                    }

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::hdf5::get_extent;

                        return B::can_load(ar)
                            && ar.is_data("timeseries/data")
                            && ar.is_attribute("timeseries/data/@binsize")
                            && ar.is_attribute("timeseries/data/@maxbinnum")
                            && get_extent(T()).size() + 1 == ar.dimensions("timeseries/data")
                        ;
                    }

                    // TODO: add functions and operators

                private:
                    std::size_t m_mn_max_number;
                    typename B::count_type m_mn_elements_in_bin;
                    std::vector<typename mean_type<B>::type> m_mn_bins;
            };

            template<typename B> class BaseWrapper<max_num_binning_tag, B> : public B {
                public:
                    virtual bool has_max_num_binning() const = 0;
            };

            template<typename T, typename B> class ResultTypeWrapper<T, max_num_binning_tag, B> : public B {
                public:
                    virtual typename max_num_binning_type<B>::type max_num_binning() const = 0;
            };

            template<typename T, typename B> class DerivedWrapper<T, max_num_binning_tag, B> : public B {
                public:
                    DerivedWrapper(): B() {}
                    DerivedWrapper(T const & arg): B(arg) {}

                    bool has_max_num_binning() const { return has_feature<T, max_num_binning_tag>::type::value; }

                    typename max_num_binning_type<B>::type max_num_binning() const { return detail::max_num_binning_impl(this->m_data); }
            };

        }
    }
}

 #endif
