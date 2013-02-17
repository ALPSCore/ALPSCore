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

#ifndef ALPS_NGS_MCRESULT_IMPL_BASE_IPP
#define ALPS_NGS_MCRESULT_IMPL_BASE_IPP

#include <alps/ngs/lib/mcresult_impl_derived.ipp>

#ifdef ALPS_HAVE_MPI
    #include <boost/mpi.hpp>
#endif

namespace alps {

    namespace detail {

        class mcresult_impl_base {

            public:

                virtual ~mcresult_impl_base() {}

                virtual bool can_rebin() const = 0;

                virtual bool jackknife_valid() const = 0;

                virtual uint64_t count() const = 0;

                virtual uint64_t bin_size() const = 0;

                virtual uint64_t max_bin_number() const = 0;

                virtual std::size_t bin_number() const = 0;
                
                template <typename T> bool is_type() const {
                    return dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const *>(this) != NULL;
                }

                template <typename T> std::vector<T> const & bins() const {
                    return dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const &>(*this).bins();
                }

                template <typename T> T const & mean() const {
                    return dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const &>(*this).mean();
                }

                template <typename T> T const & error() const {
                    return dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const &>(*this).error();
                }

                virtual bool has_variance() const = 0;

                template <typename T> T const & variance() const {
                    return dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const &>(*this).variance();
                }

                virtual bool has_tau() const = 0;

                template <typename T> T const & tau() const {
                    return dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const &>(*this).tau();
                }

                template <typename T> typename covariance_type<T>::type covariance(mcresult_impl_base const & arg) const {
                    return dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const &>(*this).covariance(
                        dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const &>(arg)
                    );
                }
                
                template <typename T> typename covariance_type<T>::type accurate_covariance(mcresult_impl_base const & arg) const {
                    return dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const &>(*this).accurate_covariance(
                        dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const &>(arg)
                    );
                }
                
                #define ALPS_NGS_MCRESULT_IMPL_BASE_OPERATOR(NAME)                                                                             \
                    template <typename T> typename boost::disable_if<                                                                          \
                        boost::is_same<T, mcresult_impl_base *>                                                                                \
                    >::type NAME ## _assign(T const & rhs) {                                                                                   \
                        if (dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const *>(this))                                          \
                            dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> &>(*this). NAME ## _assign(rhs);                         \
                        else                                                                                                                   \
                            dynamic_cast<typename boost::mpl::if_<                                                                               \
                                /* prevent windows from compiling mcresult_impl_derived<B, std::vector<std::vector<X> > >  */                   \
                                  is_std_vector<T>                                                                                               \
                                , mcresult_impl_derived<mcresult_impl_base, T>                                                                   \
                                , mcresult_impl_derived<mcresult_impl_base, std::vector<T> >                                                   \
                            >::type &>(*this). NAME ## _assign(rhs);                                                                           \
                    }                                                                                                                          \
                                                                                                                                               \
                    template <typename T> typename boost::enable_if<                                                                           \
                        boost::is_same<T, mcresult_impl_base *>                                                                                \
                    >::type NAME ## _assign(T const & rhs) {                                                                                   \
                        return NAME ## _assign_virtual(rhs);                                                                                   \
                    }                                                                                                                          \
                                                                                                                                               \
                    virtual void NAME ## _assign_virtual(mcresult_impl_base const * rhs) = 0;                                                  \
                                                                                                                                               \
                    template <typename T> typename boost::disable_if<                                                                          \
                        boost::is_same<T, mcresult_impl_base *>, mcresult_impl_base *                                                          \
                    >::type NAME (T const & rhs) const {                                                                                       \
                        if (dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const *>(this))                                          \
                            return dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const &>(*this). NAME (rhs);                      \
                        else                                                                                                                   \
                            return dynamic_cast<typename boost::mpl::if_<                                                                       \
                                /* prevent windows from compiling mcresult_impl_derived<B, std::vector<std::vector<X> > >  */                   \
                                  is_std_vector<T>                                                                                               \
                                , mcresult_impl_derived<mcresult_impl_base, T>                                                                   \
                                , mcresult_impl_derived<mcresult_impl_base, std::vector<T> >                                                   \
                            >::type const &>(*this). NAME (rhs);                                                                               \
                    }                                                                                                                          \
                    template <typename T> typename boost::enable_if<                                                                           \
                        boost::is_same<T, mcresult_impl_base *>, mcresult_impl_base *                                                          \
                    >::type NAME (T const & rhs) const {                                                                                       \
                        return NAME ## _virtual(rhs);                                                                                          \
                    }                                                                                                                          \
                                                                                                                                               \
                    virtual mcresult_impl_base * NAME ## _virtual(mcresult_impl_base const * rhs) const = 0;                                   \
                                                                                                                                               \
                    template <typename T> mcresult_impl_base * NAME ## _inverse(T const & lhs) {                                               \
                        if (dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> const *>(this))                                          \
                            return dynamic_cast<mcresult_impl_derived<mcresult_impl_base, T> &>(*this). NAME ## _inverse(lhs);                 \
                        else                                                                                                                   \
                            return dynamic_cast<typename boost::mpl::if_<                                                                       \
                                /* prevent windows from compiling mcresult_impl_derived<B, std::vector<std::vector<X> > >  */                   \
                                  is_std_vector<T>                                                                                               \
                                , mcresult_impl_derived<mcresult_impl_base, T>                                                                   \
                                , mcresult_impl_derived<mcresult_impl_base, std::vector<T> >                                                   \
                            >::type    &>(*this). NAME ## _inverse(lhs);                                                                           \
                    }
                ALPS_NGS_MCRESULT_IMPL_BASE_OPERATOR(add)
                ALPS_NGS_MCRESULT_IMPL_BASE_OPERATOR(sub)
                ALPS_NGS_MCRESULT_IMPL_BASE_OPERATOR(mul)
                ALPS_NGS_MCRESULT_IMPL_BASE_OPERATOR(div)
                #undef ALPS_NGS_MCRESULT_IMPL_BASE_OPERATOR

                virtual void set_bin_size(uint64_t binsize) = 0;

                virtual void set_bin_number(uint64_t bin_number) = 0;

                virtual void save(hdf5::archive & ar) const = 0;

                virtual void load(hdf5::archive & ar) = 0;

                virtual void output(std::ostream & os) const = 0;

                #ifdef ALPS_HAVE_MPI
                    virtual mcresult_impl_base * reduce(boost::mpi::communicator const & communicator, std::size_t binnumber) = 0;
                #endif

                virtual bool operator==(mcresult_impl_base const * rhs) const = 0;
                virtual bool operator!=(mcresult_impl_base const * rhs) const = 0;

                virtual void operator+() = 0;
                virtual void operator-() = 0;

                virtual mcresult_impl_base * sin() const = 0;
                virtual mcresult_impl_base * cos() const = 0;
                virtual mcresult_impl_base * tan() const = 0;
                virtual mcresult_impl_base * sinh() const = 0;
                virtual mcresult_impl_base * cosh() const = 0;
                virtual mcresult_impl_base * tanh() const = 0;
                virtual mcresult_impl_base * asin() const = 0;
                virtual mcresult_impl_base * acos() const = 0;
                virtual mcresult_impl_base * atan() const = 0;
// asinh, aconsh and atanh are not part of C++03 standard
//                virtual mcresult_impl_base * asinh() const = 0;
//                virtual mcresult_impl_base * acosh() const = 0;
//                virtual mcresult_impl_base * atanh() const = 0;
                virtual mcresult_impl_base * abs() const = 0;
                virtual mcresult_impl_base * sq() const = 0;
                virtual mcresult_impl_base * cb() const = 0;
                virtual mcresult_impl_base * sqrt() const = 0;
                virtual mcresult_impl_base * cbrt() const = 0;
                virtual mcresult_impl_base * exp() const = 0;
                virtual mcresult_impl_base * log() const = 0;

                virtual mcresult_impl_base * pow(double exponent) const = 0;

        };
    }
}

#endif
