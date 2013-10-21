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

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/mcresult.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/numeric/array.hpp>
#include <alps/ngs/lib/mcresult_impl_base.ipp>
#include <alps/ngs/lib/mcresult_impl_derived.ipp>

// #ifdef ALPS_NGS_USE_NEW_ALEA
//     #include <alps/ngs/alea/accumulator_set.hpp>
// #endif

#include <alps/alea/observable.h>
#include <alps/alea/abstractsimpleobservable.h>

#include <iostream>
#include <stdexcept>

namespace alps {

    mcresult::mcresult()
        : impl_(NULL) 
    {}

    mcresult::mcresult(Observable const * obs) {
        construct(obs);
    }
    
// #ifdef ALPS_NGS_USE_NEW_ALEA
//     template<typename T> bool has_cast(alps::accumulator::detail::accumulator_wrapper const & acc_wrapper) { 
//         try { 
//             acc_wrapper.get<T>();
//             return true;
//         }  catch(std::bad_cast)  { 
//             return false;
//         }
//     }
    
//     mcresult::mcresult(alps::accumulator::detail::accumulator_wrapper const & acc_wrapper) {
//         //------------------- find out value_type  -------------------
//         if(has_cast<double>(acc_wrapper))
//             impl_ = new detail::mcresult_impl_derived<detail::mcresult_impl_base, double>(acc_wrapper);
//         else if(has_cast<std::vector<double> >(acc_wrapper))
//            impl_ = new detail::mcresult_impl_derived<detail::mcresult_impl_base, std::vector<double> >(acc_wrapper);
// /*
//         else if(has_cast<boost::array<double, 3> >(acc_wrapper))
//             impl_ = new detail::mcresult_impl_derived<detail::mcresult_impl_base, boost::array<double, 3> >(acc_wrapper);
// */
//         else
//             throw std::runtime_error("type not supported in mcresult-constructor" + ALPS_STACKTRACE);
//         ref_cnt_[impl_] = 1;
//     }
// #endif

    mcresult::mcresult(mcresult const & rhs) {
        ++ref_cnt_[impl_ = rhs.impl_];
    }

    mcresult::mcresult(mcobservable const & obs) {
        construct(obs.get_impl());
    }

    mcresult::~mcresult() {
        if (impl_ && !--ref_cnt_[impl_])
            delete impl_;
    }

    mcresult & mcresult::operator=(mcresult rhs) {
        if (impl_ && !--ref_cnt_[impl_])
            delete impl_;
        ++ref_cnt_[impl_ = rhs.impl_];
        return *this;
    }
    
    #define ALPS_MCRESULT_TPL_IMPL(T)                                                                                               \
        template<> ALPS_DECL bool mcresult::is_type< T >() const { return impl_->is_type< T >(); }                                  \
        template<> ALPS_DECL std::vector< T > const & mcresult::bins< T >() const { return impl_->bins< T >(); }                    \
        template<> ALPS_DECL T const & mcresult::mean< T >() const { return impl_->mean< T >(); }                                   \
        template<> ALPS_DECL T const & mcresult::error< T >() const { return impl_->error< T >(); }                                 \
        template<> ALPS_DECL T const & mcresult::variance< T >() const { return impl_->variance< T >(); }                           \
        template<> ALPS_DECL T const & mcresult::tau< T >() const { return impl_->tau< T >(); }                                     \
        template<> ALPS_DECL covariance_type<T>::type mcresult::covariance< T >(mcresult const & arg) const {                       \
            return impl_->covariance< T >(*arg.impl_);                                                                              \
        }                                                        \
        template<> ALPS_DECL covariance_type<T>::type mcresult::accurate_covariance< T >(mcresult const & arg) const {                       \
            return impl_->accurate_covariance< T >(*arg.impl_);                                                                           \
        }                                                        \
        template<> ALPS_DECL mcresult & mcresult::operator+=< T >( T const & rhs) { impl_->add_assign(rhs); return *this; }         \
        template<> ALPS_DECL mcresult & mcresult::operator-=< T >( T const & rhs) { impl_->sub_assign(rhs); return *this; }         \
        template<> ALPS_DECL mcresult & mcresult::operator*=< T >( T const & rhs) { impl_->mul_assign(rhs); return *this; }         \
        template<> ALPS_DECL mcresult & mcresult::operator/=< T >( T const & rhs) { impl_->div_assign(rhs); return *this; }
    ALPS_MCRESULT_TPL_IMPL(double)
    ALPS_MCRESULT_TPL_IMPL(std::vector<double>)
    #undef ALPS_MCRESULT_TPL_IMPL

    #define ALPS_NGS_MCRESULT_OPERATOR_IMPL(OP, NAME)         \
        mcresult & mcresult:: OP (mcresult const & rhs) {     \
            impl_-> NAME ## _assign (rhs.impl_);              \
            return *this;                                     \
        }
    ALPS_NGS_MCRESULT_OPERATOR_IMPL(operator+=, add)
    ALPS_NGS_MCRESULT_OPERATOR_IMPL(operator-=, sub)
    ALPS_NGS_MCRESULT_OPERATOR_IMPL(operator*=, mul)
    ALPS_NGS_MCRESULT_OPERATOR_IMPL(operator/=, div)
    #undef ALPS_NGS_MCRESULT_OPERATOR_IMPL

    bool mcresult::can_rebin() const {
        return impl_->can_rebin();
    }

    bool mcresult::jackknife_valid() const {
        return impl_->jackknife_valid();
    }

    uint64_t mcresult::count() const {
        return impl_->count();
    }

    uint64_t mcresult::bin_size() const {
        return impl_->bin_size();
    }

    uint64_t mcresult::max_bin_number() const {
        return impl_->max_bin_number();
    }

    std::size_t mcresult::bin_number() const {
        return impl_->bin_number();
    }

    bool mcresult::has_variance() const {
        return impl_->has_variance();
    }

    bool mcresult::has_tau() const {
        return impl_->has_tau();
    }

    void mcresult::set_bin_size(uint64_t binsize) {
        impl_->set_bin_size(binsize);
    }

    void mcresult::set_bin_number(uint64_t bin_number) {
        impl_->set_bin_number(bin_number);
    }

    void mcresult::save(hdf5::archive & ar) const {
        impl_->save(ar);
    }

    void mcresult::load(hdf5::archive & ar) {
        impl_->save(ar);
    }

    void mcresult::output(std::ostream & os) const {
        impl_->output(os);
    }

    #ifdef ALPS_HAVE_MPI
        mcresult mcresult::reduce(boost::mpi::communicator const & communicator, std::size_t binnumber) {
            mcresult lhs;
            detail::mcresult_impl_base * impl = impl_->reduce(communicator, binnumber);
            if (communicator.rank() == 0)
                ref_cnt_[lhs.impl_ = impl] = 1;
            return lhs;
        }
    #endif

    bool mcresult::operator== (mcresult const & rhs) const {
        return impl_->operator== (rhs.impl_);
    }
    bool mcresult::operator!= (mcresult const & rhs) const {
        return impl_->operator!= (rhs.impl_);
    }

    mcresult & mcresult::operator+() {
        impl_->operator-();
        return *this;
    }
    mcresult & mcresult::operator-() {
        impl_->operator-();
        return *this;
    }

    void mcresult::construct(Observable const * obs) {
        if (dynamic_cast<AbstractSimpleObservable<double> const *>(obs) != NULL)
            impl_ = new detail::mcresult_impl_derived<detail::mcresult_impl_base, double>(
                dynamic_cast<AbstractSimpleObservable<double> const &>(*obs)
            );
        else if (dynamic_cast<AbstractSimpleObservable<std::valarray<double> > const *>(obs) != NULL)
            impl_ = new detail::mcresult_impl_derived<detail::mcresult_impl_base, std::vector<double> >(
                dynamic_cast<AbstractSimpleObservable<std::valarray<double> > const &>(*obs)
            );
        else
            throw std::runtime_error("unknown observable type" + ALPS_STACKTRACE);
        ref_cnt_[impl_] = 1;
    }
    

    std::map<detail::mcresult_impl_base *, std::size_t> mcresult::ref_cnt_;

    std::ostream & operator<<(std::ostream & os, mcresult const & res) {
        res.output(os);
        return os;
    }

    #define ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(FUN_NAME)                 \
        mcresult FUN_NAME (mcresult rhs) {                               \
            mcresult lhs;                                                \
            lhs.ref_cnt_[lhs.impl_ = rhs.impl_-> FUN_NAME ()] = 1;       \
            return lhs;                                                  \
        }
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(sin)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(cos)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(tan)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(sinh)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(cosh)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(tanh)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(asin)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(acos)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(atan)
// asinh, aconsh and atanh are not part of C++03 standard
//    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(asinh)
//    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(acosh)
//    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(atanh)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(abs)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(sq)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(cb)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(sqrt)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(cbrt)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(exp)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(log)
    #undef ALPS_NGS_MCRESULT_FREE_UNITARY_FUN

    mcresult pow(mcresult rhs, double exponent) {
        mcresult lhs;
        lhs.ref_cnt_[lhs.impl_ = rhs.impl_->pow(exponent)] = 1;
        return lhs;
    }

    #define ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_IMPL(T, OP, NAME)                  \
        mcresult OP(mcresult const & lhs, T const & rhs) {                         \
            mcresult res;                                                          \
            res.ref_cnt_[res.impl_ = lhs.impl_-> NAME (rhs)] = 1;                  \
            return res;                                                            \
        }                                                                          \
        mcresult OP(T const & lhs, mcresult const & rhs) {                         \
            mcresult res;                                                          \
            res.ref_cnt_[res.impl_ = rhs.impl_-> NAME ## _inverse (lhs)] = 1;      \
            return res;                                                            \
        }
    #define ALPS_NGS_MCRESULT_FREE_OPERATOR_IMPL(OP, NAME)                         \
        ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_IMPL(double, OP, NAME)                 \
        ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_IMPL(std::vector<double>, OP, NAME)    \
        mcresult OP (mcresult const & lhs, mcresult const & rhs) {                 \
            mcresult res;                                                          \
            res.ref_cnt_[res.impl_ = lhs.impl_-> NAME (rhs.impl_)] = 1;            \
            return res;                                                            \
        }
    ALPS_NGS_MCRESULT_FREE_OPERATOR_IMPL(operator+, add)
    ALPS_NGS_MCRESULT_FREE_OPERATOR_IMPL(operator-, sub)
    ALPS_NGS_MCRESULT_FREE_OPERATOR_IMPL(operator*, mul)
    ALPS_NGS_MCRESULT_FREE_OPERATOR_IMPL(operator/, div)
    #undef ALPS_MCRESULT_OPERATOR_IMPL
    #undef ALPS_NGS_MCRESULT_FREE_OPERATOR_TPL_IMPL

}
