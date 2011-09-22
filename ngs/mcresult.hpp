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

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/config.hpp>
#include <alps/ngs/mcobservable.hpp>

#include <alps/alea/observable_fwd.hpp>

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

    class mcresult;

    mcresult sin(mcresult rhs);
    mcresult cos(mcresult rhs);
    mcresult tan(mcresult rhs);
    mcresult sinh(mcresult rhs);
    mcresult cosh(mcresult rhs);
    mcresult tanh(mcresult rhs);
    mcresult asin(mcresult rhs);
    mcresult acos(mcresult rhs);
    mcresult atan(mcresult rhs);
    mcresult asinh(mcresult rhs);
    mcresult acosh(mcresult rhs);
    mcresult atanh(mcresult rhs);
    mcresult abs(mcresult rhs);
    mcresult sq(mcresult rhs);
    mcresult cb(mcresult rhs);
    mcresult sqrt(mcresult rhs);
    mcresult cbrt(mcresult rhs);
    mcresult exp(mcresult rhs);
    mcresult log(mcresult rhs);

    mcresult pow(mcresult rhs, double exponent);
    
    #define ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_DECL(OP, T)                             \
        mcresult OP(mcresult const & lhs, T const & rhs);                               \
        mcresult OP( T const & lhs, mcresult const & rhs);
    #define ALPS_NGS_MCRESULT_FREE_OPERATOR_DECL(OP)                                    \
        ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_DECL(OP, double)                            \
        ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_DECL(OP, std::vector<double>)               \
        mcresult OP (mcresult const & lhs, mcresult const & rhs);
    ALPS_NGS_MCRESULT_FREE_OPERATOR_DECL(operator+)
    ALPS_NGS_MCRESULT_FREE_OPERATOR_DECL(operator-)
    ALPS_NGS_MCRESULT_FREE_OPERATOR_DECL(operator*)
    ALPS_NGS_MCRESULT_FREE_OPERATOR_DECL(operator/)
    #undef ALPS_NGS_MCRESULT_FREE_OPERATOR_DECL
    #undef ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_DECL

    class mcresult {

        public:

            friend mcresult sin(mcresult rhs);
            friend mcresult cos(mcresult rhs);
            friend mcresult tan(mcresult rhs);
            friend mcresult sinh(mcresult rhs);
            friend mcresult cosh(mcresult rhs);
            friend mcresult tanh(mcresult rhs);
            friend mcresult asin(mcresult rhs);
            friend mcresult acos(mcresult rhs);
            friend mcresult atan(mcresult rhs);
            friend mcresult asinh(mcresult rhs);
            friend mcresult acosh(mcresult rhs);
            friend mcresult atanh(mcresult rhs);
            friend mcresult abs(mcresult rhs);
            friend mcresult sq(mcresult rhs);
            friend mcresult cb(mcresult rhs);
            friend mcresult sqrt(mcresult rhs);
            friend mcresult cbrt(mcresult rhs);
            friend mcresult exp(mcresult rhs);
            friend mcresult log(mcresult rhs);

            friend mcresult pow(mcresult rhs, double exponent);

            #define ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_FRIEND(OP, T)                            \
                friend mcresult OP(mcresult const & lhs, T const & rhs);                         \
                friend mcresult OP( T const & lhs, mcresult const & rhs);
            #define ALPS_NGS_MCRESULT_FREE_OPERATOR_FRIEND(OP)                                   \
                ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_FRIEND(OP, double)                           \
                ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_FRIEND(OP, std::vector<double>)              \
                friend mcresult OP (mcresult const & lhs, mcresult const & rhs);
            ALPS_NGS_MCRESULT_FREE_OPERATOR_FRIEND(operator+)
            ALPS_NGS_MCRESULT_FREE_OPERATOR_FRIEND(operator-)
            ALPS_NGS_MCRESULT_FREE_OPERATOR_FRIEND(operator*)
            ALPS_NGS_MCRESULT_FREE_OPERATOR_FRIEND(operator/)
            #undef ALPS_NGS_MCRESULT_FREE_OPERATOR_FRIEND
            #undef ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_FRIEND

            mcresult();
            mcresult(Observable const * obs);
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

            template <typename T> T const & covariance() const;

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

    std::ostream & operator<<(std::ostream & os, mcresult const & res);

}

#endif
