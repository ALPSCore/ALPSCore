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

#ifndef ALPS_NGS_MCRESULT_HPP
#define ALPS_NGS_MCRESULT_HPP

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/config.hpp>
#include <alps/ngs/mcobservable.hpp>

// #ifdef ALPS_NGS_USE_NEW_ALEA
//     #include <alps/ngs/alea/accumulator_set.hpp>
// #endif
#include <alps/alea/observable_fwd.hpp>
#include <alps/type_traits/covariance_type.hpp>

#ifdef ALPS_HAVE_MPI
    #include <boost/mpi.hpp>
#endif

#include <map>
#include <vector>
#include <iostream>

namespace alps {

    namespace detail {

        class mcresult_impl_base;

    }

    class ALPS_DECL mcresult;

    ALPS_DECL mcresult sin(mcresult rhs);
    ALPS_DECL mcresult cos(mcresult rhs);
    ALPS_DECL mcresult tan(mcresult rhs);
    ALPS_DECL mcresult sinh(mcresult rhs);
    ALPS_DECL mcresult cosh(mcresult rhs);
    ALPS_DECL mcresult tanh(mcresult rhs);
    ALPS_DECL mcresult asin(mcresult rhs);
    ALPS_DECL mcresult acos(mcresult rhs);
    ALPS_DECL mcresult atan(mcresult rhs);
// asinh, aconsh and atanh are not part of C++03 standard
//    ALPS_DECL mcresult asinh(mcresult rhs);
//    ALPS_DECL mcresult acosh(mcresult rhs);
//    ALPS_DECL mcresult atanh(mcresult rhs);
    ALPS_DECL mcresult abs(mcresult rhs);
    ALPS_DECL mcresult sq(mcresult rhs);
    ALPS_DECL mcresult cb(mcresult rhs);
    ALPS_DECL mcresult sqrt(mcresult rhs);
    ALPS_DECL mcresult cbrt(mcresult rhs);
    ALPS_DECL mcresult exp(mcresult rhs);
    ALPS_DECL mcresult log(mcresult rhs);

    ALPS_DECL mcresult pow(mcresult rhs, double exponent);
    
    #define ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_DECL(OP, T)                             \
        ALPS_DECL mcresult OP(mcresult const & lhs, T const & rhs);                     \
        ALPS_DECL mcresult OP( T const & lhs, mcresult const & rhs);
    #define ALPS_NGS_MCRESULT_FREE_OPERATOR_DECL(OP)                                    \
        ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_DECL(OP, double)                  \
        ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_DECL(OP, std::vector<double>)     \
        ALPS_DECL mcresult OP (mcresult const & lhs, mcresult const & rhs);
    ALPS_NGS_MCRESULT_FREE_OPERATOR_DECL(operator+)
    ALPS_NGS_MCRESULT_FREE_OPERATOR_DECL(operator-)
    ALPS_NGS_MCRESULT_FREE_OPERATOR_DECL(operator*)
    ALPS_NGS_MCRESULT_FREE_OPERATOR_DECL(operator/)
    #undef ALPS_NGS_MCRESULT_FREE_OPERATOR_DECL
    #undef ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_DECL

    class ALPS_DECL mcresult {

        public:

            friend ALPS_DECL mcresult sin(mcresult rhs);
            friend ALPS_DECL mcresult cos(mcresult rhs);
            friend ALPS_DECL mcresult tan(mcresult rhs);
            friend ALPS_DECL mcresult sinh(mcresult rhs);
            friend ALPS_DECL mcresult cosh(mcresult rhs);
            friend ALPS_DECL mcresult tanh(mcresult rhs);
            friend ALPS_DECL mcresult asin(mcresult rhs);
            friend ALPS_DECL mcresult acos(mcresult rhs);
            friend ALPS_DECL mcresult atan(mcresult rhs);
            friend ALPS_DECL mcresult asinh(mcresult rhs);
            friend ALPS_DECL mcresult acosh(mcresult rhs);
            friend ALPS_DECL mcresult atanh(mcresult rhs);
            friend ALPS_DECL mcresult abs(mcresult rhs);
            friend ALPS_DECL mcresult sq(mcresult rhs);
            friend ALPS_DECL mcresult cb(mcresult rhs);
            friend ALPS_DECL mcresult sqrt(mcresult rhs);
            friend ALPS_DECL mcresult cbrt(mcresult rhs);
            friend ALPS_DECL mcresult exp(mcresult rhs);
            friend ALPS_DECL mcresult log(mcresult rhs);

            friend ALPS_DECL mcresult pow(mcresult rhs, double exponent);

            #define ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_FRIEND(OP, T)                            \
                friend ALPS_DECL mcresult OP(mcresult const & lhs, T const & rhs);                         \
                friend ALPS_DECL mcresult OP( T const & lhs, mcresult const & rhs);
            #define ALPS_NGS_MCRESULT_FREE_OPERATOR_FRIEND(OP)                                   \
                ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_FRIEND(OP, double)                           \
                ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_FRIEND(OP, std::vector<double>)              \
                friend ALPS_DECL mcresult OP (mcresult const & lhs, mcresult const & rhs);
            ALPS_NGS_MCRESULT_FREE_OPERATOR_FRIEND(operator+)
            ALPS_NGS_MCRESULT_FREE_OPERATOR_FRIEND(operator-)
            ALPS_NGS_MCRESULT_FREE_OPERATOR_FRIEND(operator*)
            ALPS_NGS_MCRESULT_FREE_OPERATOR_FRIEND(operator/)
            #undef ALPS_NGS_MCRESULT_FREE_OPERATOR_FRIEND
            #undef ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_FRIEND

            mcresult();
            mcresult(Observable const * obs);
            // #ifdef ALPS_NGS_USE_NEW_ALEA
            //     mcresult(alps::accumulator::detail::accumulator_wrapper const & acc_wrapper);
            // #endif
            mcresult(mcresult const & rhs);
            mcresult(mcobservable const & rhs);

            virtual ~mcresult();

            mcresult & operator=(mcresult rhs);

            bool can_rebin() const;
            bool jackknife_valid() const;

            uint64_t count() const;

            uint64_t bin_size() const;
            uint64_t max_bin_number() const;
            std::size_t bin_number() const;

            template <typename T> bool is_type() const;

            template <typename T> std::vector<T> const & bins() const;

            template <typename T> T const & mean() const;

            template <typename T> T const & error() const;

            bool has_variance() const;
            template <typename T> T const & variance() const;

            bool has_tau() const;
            template <typename T> T const & tau() const;

            template <typename T> typename covariance_type<T>::type covariance(mcresult const &) const;
            template <typename T> typename covariance_type<T>::type accurate_covariance(mcresult const &) const;

            #define ALPS_NGS_MCRESULT_ASSIGN_OPERATORS(OP)                                       \
                template <typename T> mcresult & OP (T const & rhs);                             \
                mcresult & OP (mcresult const & rhs);
            ALPS_NGS_MCRESULT_ASSIGN_OPERATORS(operator+=)
            ALPS_NGS_MCRESULT_ASSIGN_OPERATORS(operator-=)
            ALPS_NGS_MCRESULT_ASSIGN_OPERATORS(operator*=)
            ALPS_NGS_MCRESULT_ASSIGN_OPERATORS(operator/=)
            #undef ALPS_NGS_MCRESULT_ASSIGN_OPERATORS

            void set_bin_size(uint64_t binsize);
            void set_bin_number(uint64_t bin_number);

            void save(hdf5::archive & ar) const;
            void load(hdf5::archive & ar);

            void output(std::ostream & os) const;

            #ifdef ALPS_HAVE_MPI
                mcresult reduce(boost::mpi::communicator const & communicator, std::size_t binnumber);
            #endif

            bool operator==(mcresult const & rhs) const;
            bool operator!=(mcresult const & rhs) const;
            mcresult & operator+();
            mcresult & operator-();

        private:

            void construct(Observable const * obs);

            detail::mcresult_impl_base * impl_;
            static std::map<detail::mcresult_impl_base *, std::size_t> ref_cnt_;

    };

    ALPS_DECL std::ostream & operator<<(std::ostream & os, mcresult const & res);

}

#endif
