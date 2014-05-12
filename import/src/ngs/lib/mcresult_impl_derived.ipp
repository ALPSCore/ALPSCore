/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_MCRESULT_IMPL_DERIVED_IPP
#define ALPS_NGS_MCRESULT_IMPL_DERIVED_IPP

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/boost_mpi.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/short_print.hpp>

#include <alps/alea/mcdata.hpp>

// #ifdef ALPS_NGS_USE_NEW_ALEA
//     #include <alps/ngs/alea.hpp>
// #endif

#ifdef ALPS_HAVE_MPI
    #include <boost/mpi.hpp>
#endif

#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace alps {

    namespace detail {

        template<typename T> struct is_std_vector : boost::false_type {};
        template<typename T> struct is_std_vector<std::vector<T> > : boost::true_type {};

        template<typename B, typename T> class mcresult_impl_derived : public B, public alea::mcdata<T> {

            public:

                template<typename U> mcresult_impl_derived(AbstractSimpleObservable<U> const & obs)
                    : alea::mcdata<T>(obs) 
                {}
                
                // #ifdef ALPS_NGS_USE_NEW_ALEA
                // //TODO4 just pass the result_type wrapper to mcdata<..>..
                // mcresult_impl_derived(accumulator::detail::accumulator_wrapper const & acc_wrapper)
                //     : alea::mcdata<T>(acc_wrapper.get<T>())
                // {
                // }
                // #endif
                
                mcresult_impl_derived(alea::mcdata<T> const & data)
                    : alea::mcdata<T>(data) 
                {}

                mcresult_impl_derived(
                      int64_t count
                    , T const & mean
                    , T const & error
                    , boost::optional<T> const & variance_opt
                    , boost::optional<T> const & tau_opt
                    , uint64_t binsize
                    , uint64_t max_bin_number
                    , std::vector<T> const & values
                ): alea::mcdata<T>(
                    count, mean, error, variance_opt, tau_opt, binsize, max_bin_number, values
                ) {}

                bool can_rebin() const {
                    return alea::mcdata<T>::can_rebin();
                }

                bool jackknife_valid() const {
                    return alea::mcdata<T>::jackknife_valid();
                }

                uint64_t count() const {
                    return alea::mcdata<T>::count();
                }

                uint64_t bin_size() const {
                    return alea::mcdata<T>::bin_size();
                }

                uint64_t max_bin_number() const {
                    return alea::mcdata<T>::max_bin_number();
                }

                std::size_t bin_number() const {
                    return alea::mcdata<T>::bin_number();
                }

                std::vector<T> const & bins() const {
                    return alea::mcdata<T>::bins();
                }

                T const & mean() const {
                    return alea::mcdata<T>::mean();
                }

                T const & error() const {
                    return alea::mcdata<T>::error();
                }

                bool has_variance() const {
                    return alea::mcdata<T>::has_variance();
                }

                T const & variance() const {
                    return alea::mcdata<T>::variance();
                }

                bool has_tau() const {
                    return alea::mcdata<T>::has_tau();
                }

                T const & tau() const {
                    return alea::mcdata<T>::tau();
                }

				typename ::alps::covariance_type<T>::type covariance(mcresult_impl_derived<B, T> const & arg) const {
                    return alea::mcdata<T>::covariance(static_cast<alea::mcdata<T> const &>(arg));
                }

				typename ::alps::covariance_type<T>::type accurate_covariance(mcresult_impl_derived<B, T> const & arg) const {
                    return alea::mcdata<T>::accurate_covariance(static_cast<alea::mcdata<T> const &>(arg));
                }
                #define ALPS_NGS_MCRESULT_IMPL_DERIVED_OPERATOR(NAME, OP, OP_ASSIGN)                                                           \
                    template <typename U> typename boost::enable_if<                                                                           \
                          typename boost::is_same<T, U >::type                                                                                 \
                      /*, typename boost::is_same<typename alea::mcdata<T>::element_type, U>::type*/                                           \
                    >::type NAME ## _assign (U const & rhs) {                                                                                  \
                        static_cast<alea::mcdata<T> &>(*this) OP_ASSIGN rhs;                                                                   \
                    }                                                                                                                          \
                                                                                                                                               \
                    template <typename U> typename boost::disable_if<                                                                          \
                          typename boost::is_same<T, U >::type                                                                                 \
                      /*, typename boost::is_same<typename alea::mcdata<T>::element_type, U>::type*/                                           \
                    >::type NAME ## _assign (U const & rhs) {                                                                                  \
                        throw std::runtime_error("Invalid cast" + ALPS_STACKTRACE);                                                            \
                    }                                                                                                                          \
                                                                                                                                               \
                    void NAME ## _assign_virtual (B const * rhs) {                                                                             \
                        static_cast<alea::mcdata<T> &>(*this)                                                                                  \
                            OP_ASSIGN static_cast<alea::mcdata<T> const &>(*dynamic_cast<mcresult_impl_derived<B, T> const *>(rhs));           \
                    }                                                                                                                          \
                                                                                                                                               \
                                                                                                                                               \
                    template <typename U> typename boost::enable_if<typename boost::mpl::or_<                                                  \
                          typename boost::is_same<T, U>::type                                                                                  \
                      /*, typename boost::is_same<typename alea::mcdata<T>::element_type, U>::type*/                                           \
                    >::type, B *>::type NAME (U const & rhs) const {                                                                           \
                        return new mcresult_impl_derived<B, T>(                                                                                \
                            static_cast<alea::mcdata<T> const &>(*this) OP rhs                                                                 \
                        );                                                                                                                     \
                    }                                                                                                                          \
                                                                                                                                               \
                    template <typename U> typename boost::disable_if<typename boost::mpl::or_<                                                 \
                          typename boost::is_same<T, U>::type                                                                                  \
                      /* , typename boost::is_same<typename alea::mcdata<T>::element_type, U>::type*/                                          \
                    >::type, B *>::type NAME (U const & rhs) const {                                                                           \
                        throw std::runtime_error("Invalid cast" + ALPS_STACKTRACE);                                                            \
                        return NULL;                                                                                                           \
                    }                                                                                                                          \
                                                                                                                                               \
                    template<typename U> typename boost::enable_if<is_std_vector<U>, B *>::type NAME ## _virtual_impl (B const * rhs) const {  \
                        if (dynamic_cast<mcresult_impl_derived<B, U> const *>(rhs))                                                            \
                            return new mcresult_impl_derived<B, U>(                                                                            \
                                  static_cast<alea::mcdata<U> const &>(*this)                                                                  \
                                OP static_cast<alea::mcdata<U> const &>(dynamic_cast<mcresult_impl_derived<B, U> const &>(*rhs))               \
                            );                                                                                                                 \
                        else if (dynamic_cast<mcresult_impl_derived<B, typename alea::mcdata<U>::element_type> const *>(rhs))                  \
                            return new mcresult_impl_derived<B, U>(                                                                            \
                                   static_cast<alea::mcdata<U> const &>(*this)                                                                 \
                                OP static_cast<alea::mcdata<typename alea::mcdata<U>::element_type> const &>(                                  \
                                       dynamic_cast<mcresult_impl_derived<B, typename alea::mcdata<U>::element_type> const &>(*rhs)            \
                                   )                                                                                                           \
                            );                                                                                                                 \
                        else {                                                                                                                 \
                            throw std::runtime_error("Invalid cast" + ALPS_STACKTRACE);                                                        \
                            return new mcresult_impl_derived<B, U>(*this);                                                                     \
                        }                                                                                                                      \
                    }                                                                                                                          \
                                                                                                                                               \
                    template<typename U> typename boost::disable_if<is_std_vector<U>, B *>::type NAME ## _virtual_impl (B const * rhs) const { \
                        if (dynamic_cast<mcresult_impl_derived<B, U> const *>(rhs))                                                            \
                            return new mcresult_impl_derived<B, U>(                                                                            \
                                  static_cast<alea::mcdata<U> const &>(*this)                                                                  \
                                OP static_cast<alea::mcdata<U> const &>(dynamic_cast<mcresult_impl_derived<B, U> const &>(*rhs))               \
                            );                                                                                                                 \
                        else if (dynamic_cast<mcresult_impl_derived<B, typename alea::mcdata<U>::element_type> const *>(rhs))                  \
                            return new mcresult_impl_derived<B, U>(                                                                            \
                                   static_cast<alea::mcdata<U> const &>(*this)                                                                 \
                                OP static_cast<alea::mcdata<typename alea::mcdata<U>::element_type> const &>(                                  \
                                       dynamic_cast<mcresult_impl_derived<B, typename alea::mcdata<U>::element_type> const &>(*rhs)            \
                                   )                                                                                                           \
                            );                                                                                                                 \
                        else if (dynamic_cast<mcresult_impl_derived<B, std::vector<U> > const *>(rhs))                                         \
                            return static_cast<B *>(new mcresult_impl_derived<B, std::vector<U> >(                                             \
                                   static_cast<alea::mcdata<U> const &>(*this)                                                                 \
                                OP static_cast<alea::mcdata<std::vector<U> > const &>(                                                         \
                                       dynamic_cast<mcresult_impl_derived<B, std::vector<U> > const &>(*rhs)                                   \
                                   )                                                                                                           \
                            ));                                                                                                                \
                        else {                                                                                                                 \
                            throw std::runtime_error("Invalid cast" + ALPS_STACKTRACE);                                                        \
                            return new mcresult_impl_derived<B, U>(*this);                                                                     \
                        }                                                                                                                      \
                    }                                                                                                                          \
                                                                                                                                               \
                    B * NAME ## _virtual (B const * rhs) const {                                                                               \
                        return NAME ## _virtual_impl<T>(rhs);                                                                                  \
                    }                                                                                                                          \
                                                                                                                                               \
                    template <typename U> typename boost::enable_if<typename boost::mpl::or_<                                                  \
                          typename boost::is_same<T, U>::type                                                                                  \
                        , typename boost::mpl::and_<                                                                                           \
                              typename boost::is_scalar<U>::type                                                                               \
                            , typename boost::is_same<typename alea::mcdata<T>::element_type, U>::type                                         \
                        >::type                                                                                                                \
                    >::type, B *>::type NAME ## _inverse(U const & lhs) const {                                                                \
                        return new mcresult_impl_derived<B, T>(                                                                                \
                            lhs OP static_cast<alea::mcdata<T> const &>(*this)                                                                 \
                        );                                                                                                                     \
                    }                                                                                                                          \
                                                                                                                                               \
                    template <typename U> typename boost::disable_if<typename boost::mpl::or_<                                                 \
                          typename boost::is_same<T, U>::type                                                                                  \
                        , typename boost::mpl::and_<                                                                                           \
                              typename boost::is_scalar<U>::type                                                                               \
                            , typename boost::is_same<typename alea::mcdata<T>::element_type, U>::type                                         \
                        >::type                                                                                                                \
                    >::type, B *>::type NAME ## _inverse(U const & rhs) const {                                                                \
                        throw std::runtime_error("Invalid cast" + ALPS_STACKTRACE);                                                            \
                        return NULL;                                                                                                           \
                    }
                ALPS_NGS_MCRESULT_IMPL_DERIVED_OPERATOR(add, +, +=)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_OPERATOR(sub, -, -=)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_OPERATOR(mul, *, *=)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_OPERATOR(div, /, /=)
                #undef ALPS_NGS_MCRESULT_IMPL_DERIVED_OPERATOR

                void set_bin_size(uint64_t binsize) {
                    alea::mcdata<T>::set_bin_size(binsize);
                }

                void set_bin_number(uint64_t bin_number) {
                    alea::mcdata<T>::set_bin_number(bin_number);
                }

                void save(hdf5::archive & ar) const {
                    ar
                        << make_pvp(ar.get_context(), static_cast<alea::mcdata<T> const &>(*this))
                    ;
                }

                void load(hdf5::archive & ar) {
                    alea::mcdata<T>::save(ar);
                }

                void output(std::ostream & os) const {
                    if (alea::mcdata<T>::count() == 0)
                        os 
                            << "No Measurements" 
                        ;
                    else {
                        os 
                            << short_print(alea::mcdata<T>::mean(), 6) << "(" << count() << ") "
                            << "+/-" << short_print(alea::mcdata<T>::error(), 3) << " "
                            << short_print(alea::mcdata<T>::bins(), 4) 
                            << "#" << alea::mcdata<T>::bin_size()
                        ;
                    }
                }

                #ifdef ALPS_HAVE_MPI
                    B * reduce(boost::mpi::communicator const & communicator, std::size_t binnumber) {
                        if (communicator.rank() == 0)
                            return reduce_master(communicator, binnumber, typename boost::is_scalar<T>::type());
                        else {
                            reduce_slave(communicator, binnumber, typename boost::is_scalar<T>::type());
                            return NULL;
                        }
                    }
                #endif

                bool operator==(B const * rhs) const {
                    return static_cast<alea::mcdata<T> const &>(*this)
                        == static_cast<alea::mcdata<T> const &>(dynamic_cast<mcresult_impl_derived<B, T> const &>(*rhs))
                    ;
                }
                bool operator!=(B const * rhs) const {
                    return static_cast<alea::mcdata<T> const &>(*this)
                        != static_cast<alea::mcdata<T> const &>(dynamic_cast<mcresult_impl_derived<B, T> const &>(*rhs))
                    ;
                }

                void operator+() {
                    +static_cast<alea::mcdata<T> &>(*this);
                }
                void operator-() {
                    -static_cast<alea::mcdata<T> &>(*this);
                }

                #define ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(FUN_NAME)                                                 \
                    B * FUN_NAME () const {                                                                                       \
                        return new mcresult_impl_derived<B, T>(alea:: FUN_NAME (static_cast<alea::mcdata<T> const &>(*this)));    \
                    }
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(sin)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(cos)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(tan)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(sinh)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(cosh)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(tanh)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(asin)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(acos)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(atan)
// asinh, aconsh and atanh are not part of C++03 standard
//                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(asinh)
//                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(acosh)
//                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(atanh)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(abs)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(sq)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(cb)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(sqrt)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(cbrt)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(exp)
                ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN(log)
                #undef ALPS_NGS_MCRESULT_IMPL_DERIVED_FREE_UNITARY_FUN

                B * pow(double exponent) const {
                    return new mcresult_impl_derived<B, T>(alea::pow(static_cast<alea::mcdata<T> const &>(*this), exponent));
                }

            private:

                #ifdef ALPS_HAVE_MPI

                    B * reduce_master(
                          boost::mpi::communicator const & communicator
                        , std::size_t binnumber
                        , boost::true_type
                    ) {
                        using std::sqrt;
                        using alps::numeric::sq;
                        uint64_t global_count;
                        boost::mpi::reduce(communicator, count(), global_count, std::plus<uint64_t>(), 0);
                        std::vector<T> local(2, 0), global(alea::mcdata<T>::has_variance() ? 3 : 2, 0);
                        local[0] = alea::mcdata<T>::mean() * static_cast<double>(count());
                        local[1] = sq(alea::mcdata<T>::error()) * sq(static_cast<double>(count()));
                        if (alea::mcdata<T>::has_variance())
                            local.push_back(alea::mcdata<T>::variance() * static_cast<double>(count()));
                        boost::mpi::reduce(communicator, local, global, std::plus<double>(), 0);
                        boost::optional<T> global_variance_opt;
                        if (alea::mcdata<T>::has_variance())
                            global_variance_opt = global[2] / static_cast<double>(global_count);
                        boost::optional<T> global_tau_opt;
                        if (alea::mcdata<T>::has_tau()) {
                            T global_tau;
                            boost::mpi::reduce(communicator, alea::mcdata<T>::tau() * static_cast<double>(count()), global_tau, std::plus<double>(), 0);
                            global_tau_opt = global_tau / static_cast<double>(global_count);
                        }
                        std::vector<T> global_bins;
                        std::size_t binsize_times = 1;
                        if (alea::mcdata<T>::bin_number() > 0) {
                            std::vector<T> local_bins(binnumber);
                            binsize_times = partition_bins(local_bins, communicator);
                            global_bins.resize(local_bins.size());
                            boost::mpi::reduce(communicator, local_bins, global_bins, std::plus<T>(), 0);
                        }
                        return new mcresult_impl_derived<B, T>(
                              global_count
                            , global[0] / static_cast<double>(global_count)
                            , sqrt(global[1]) / static_cast<double>(global_count)
                            , global_variance_opt
                            , global_tau_opt
                            , binsize_times * alea::mcdata<T>::bin_size()
                            , alea::mcdata<T>::max_bin_number()
                            , global_bins
                        );
                    }

                    B * reduce_master(
                          boost::mpi::communicator const & communicator
                        , std::size_t binnumber
                        , boost::false_type
                    ) {
                        using alps::numeric::sq;
                        using alps::numeric::sqrt;
                        using boost::numeric::operators::operator*;
                        using boost::numeric::operators::operator/;
                        uint64_t global_count;
                        boost::mpi::reduce(communicator, count(), global_count, std::plus<uint64_t>(), 0);
                        T global_mean, global_error, global_variance;
                        boost::mpi::reduce(communicator, alea::mcdata<T>::mean() * static_cast<double>(count()), global_mean, std::plus<double>(), 0);
                        boost::mpi::reduce(communicator, sq(alea::mcdata<T>::error()) * sq(static_cast<double>(count())), global_error, std::plus<double>(), 0);
                        boost::optional<T> global_variance_opt;
                        if (alea::mcdata<T>::has_variance()) {
                            boost::mpi::reduce(communicator, alea::mcdata<T>::variance() * static_cast<double>(count()), global_variance, std::plus<double>(), 0);
                            global_variance_opt = global_variance / static_cast<double>(global_count);
                        }
                        boost::optional<T> global_tau_opt;
                        if (alea::mcdata<T>::has_tau()) {
                            T global_tau;
                            boost::mpi::reduce(communicator, alea::mcdata<T>::tau() * static_cast<double>(count()), global_tau, std::plus<double>(), 0);
                            global_tau_opt = global_tau / static_cast<double>(global_count);
                        }
                        std::vector<T> global_bins;
                        std::size_t binsize = 0, elementsize = alea::mcdata<T>::mean().size();
                        if (alea::mcdata<T>::bin_number() > 0) {
                            std::vector<T> local_bins(binnumber, T(elementsize));
                            binsize = partition_bins(local_bins, communicator);
                            binnumber = local_bins.size(); // partition_bins() may reduce binnumber
                            global_bins.resize(binnumber);
                            std::vector<double> local_raw_bins(binnumber * elementsize), global_raw_bins(binnumber * elementsize);
                            for (typename std::vector<T>::iterator it = local_bins.begin(); it != local_bins.end(); ++it)
                                std::copy(it->begin(), it->end(), local_raw_bins.begin() + ((it - local_bins.begin()) * elementsize));
                            boost::mpi::reduce(communicator, local_raw_bins, global_raw_bins, std::plus<double>(), 0);
                            for (typename std::vector<T>::iterator it = global_bins.begin(); it != global_bins.end(); ++it) {
                                it->resize(elementsize);
                                std::copy(global_raw_bins.begin() + (it - global_bins.begin()) * elementsize, global_raw_bins.begin() + (it - global_bins.begin() + 1) * elementsize, it->begin());
                            }
                        }
                        return new mcresult_impl_derived<B, T>(
                              global_count
                            , global_mean / static_cast<double>(global_count)
                            , sqrt(global_error) / static_cast<double>(global_count)
                            , global_variance_opt
                            , global_tau_opt
                            , elementsize * binsize
                            , alea::mcdata<T>::max_bin_number()
                            , global_bins
                        );
                    }

                    void reduce_slave(
                          boost::mpi::communicator const & communicator
                        , std::size_t binnumber
                        , boost::true_type
                    ) {
                        using alps::numeric::sq;
                        boost::mpi::reduce(communicator, count(), std::plus<uint64_t>(), 0);
                        std::vector<T> local(2, 0);
                        local[0] = alea::mcdata<T>::mean() * static_cast<double>(count());
                        local[1] = sq(alea::mcdata<T>::error()) * sq(static_cast<double>(count()));
                        if (alea::mcdata<T>::has_variance())
                            local.push_back(alea::mcdata<T>::variance() * static_cast<double>(count()));
                        boost::mpi::reduce(communicator, local, std::plus<double>(), 0);
                        if (alea::mcdata<T>::has_tau())
                            boost::mpi::reduce(communicator, alea::mcdata<T>::tau() * static_cast<double>(count()), std::plus<double>(), 0);
                        if (alea::mcdata<T>::bin_number() > 0) {
                            std::vector<T> local_bins(binnumber);
                            partition_bins(local_bins, communicator);
                            boost::mpi::reduce(communicator, local_bins, std::plus<double>(), 0);
                        }
                    }

                    void reduce_slave(
                          boost::mpi::communicator const & communicator
                        , std::size_t binnumber
                        , boost::false_type
                    ) {
                        using alps::numeric::sq;
                        using boost::numeric::operators::operator*;
                        boost::mpi::reduce(communicator, count(), std::plus<uint64_t>(), 0);
                        boost::mpi::reduce(communicator, alea::mcdata<T>::mean() * static_cast<double>(count()), std::plus<double>(), 0);
                        boost::mpi::reduce(communicator, sq(alea::mcdata<T>::error()) * sq(static_cast<double>(count())), std::plus<double>(), 0);
                        if (alea::mcdata<T>::has_variance())
                            boost::mpi::reduce(communicator, alea::mcdata<T>::variance() * static_cast<double>(count()), std::plus<double>(), 0);
                        if (alea::mcdata<T>::has_tau())
                            boost::mpi::reduce(communicator, alea::mcdata<T>::tau() * static_cast<double>(count()), std::plus<double>(), 0);
                        std::size_t elementsize = alea::mcdata<T>::mean().size();
                        if (alea::mcdata<T>::bin_number() > 0) {
                            std::vector<T> local_bins(binnumber, T(elementsize));
                            partition_bins(local_bins, communicator);
                            binnumber = local_bins.size(); // partition_bins() may reduce binnumber
                            std::vector<double> local_raw_bins(binnumber * elementsize);
                            for (typename std::vector<T>::iterator it = local_bins.begin(); it != local_bins.end(); ++it)
                                std::copy(it->begin(), it->end(), local_raw_bins.begin() + ((it - local_bins.begin()) * elementsize));
                            boost::mpi::reduce(communicator, local_raw_bins, std::plus<double>(), 0);
                        }
                    }

                    std::size_t partition_bins (std::vector<T> & bins, boost::mpi::communicator const & communicator) {
                        using boost::numeric::operators::operator+;
                        using boost::numeric::operators::operator/;
                        alea::mcdata<T>::set_bin_size(boost::mpi::all_reduce(communicator, alea::mcdata<T>::bin_size(), boost::mpi::maximum<std::size_t>()));
                        std::vector<int> buffer(2 * communicator.size()), index(communicator.size());
                        int data[2] = {communicator.rank(), static_cast<int>(alea::mcdata<T>::bin_number())};
                        boost::mpi::all_gather(communicator, data, 2, buffer);
                        for (std::vector<int>::const_iterator it = buffer.begin(); it != buffer.end(); it += 2)
                            index[*it] = *(it + 1);
                        int total_bins = std::accumulate(index.begin(), index.end(), 0);
                        if (total_bins < bins.size()) // limit binnumber to number of available bins
                            bins.resize(total_bins);
                        int perbin = total_bins / bins.size();
                        int start = std::accumulate(index.begin(), index.begin() + communicator.rank(), 0);
                        for (int i = start / perbin, j = start % perbin, k = 0; i < bins.size() && k < alea::mcdata<T>::bin_number(); ++k) {
                            bins[i] = bins[i] + alea::mcdata<T>::bins()[k] / perbin;
                            if (++j == perbin) {
                                ++i;
                                j = 0;
                            }
                        }
                        return perbin;
                    }

                #endif
        };
    }
}

#endif
