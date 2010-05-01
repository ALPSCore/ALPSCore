/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@comp-phys.org>,
*                            Beat Ammon <beat.ammon@bluewin.ch>,
*                            Andreas Laeuchli <laeuchli@comp-phys.org>,
*                            Synge Todo <wistaria@comp-phys.org>,
*                            Andreas Lange <alange@phys.ethz.ch>,
*                            Ping Nang Ma <pingnang@itp.phys.ethz.ch>
*                            Lukas Gamper <gamperl@gmail.com>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/* $Id: binned_data.h 3545 2010-03-22 10:00:00Z gamperl $ */

#ifndef ALPS_ALEA_BINNED_DATA_HPP
#define ALPS_ALEA_BINNED_DATA_HPP

#include <alps/config.h>
#include <alps/parser/parser.h>
#include <alps/alea/nan.h>
#include <alps/alea/simpleobservable.h>
#include <alps/numeric/vector_functions.hpp>
#include <alps/numeric/valarray_functions.hpp>
#include <alps/numeric/vector_valarray_conversion.hpp>
#include <alps/numeric/outer_product.hpp>
#include <alps/type_traits/change_value_type.hpp>
#include <alps/type_traits/average_type.hpp>
#include <alps/type_traits/covariance_type.hpp>
#include <alps/utility/numeric_cast.hpp>
#include <alps/utility/resize.hpp>

#include <boost/config.hpp>
#include <boost/functional.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/optional/optional.hpp>

#include <iostream>
#include <numeric>
#include <vector>

namespace alps { 
    namespace alea {
        template <class T> class binned_data {
            public:
                template <class X> friend class binned_data;
                typedef T value_type;
                typedef typename change_value_type<T,double>::type time_type;
                typedef std::size_t size_type;
                typedef double count_type;
                typedef typename average_type<T>::type result_type;
                typedef typename change_value_type<T,int>::type convergence_type;
                typedef typename change_value_type_replace_valarray<value_type,std::string>::type label_type;
                typedef typename covariance_type<T>::type covariance_type;
                binned_data()
                    , data_is_analyzed_(true)
                    , jacknife_bins_valid_(true)
                    , cannot_rebin_(false)
                {}
                template <class X, class S> binned_data(binned_data<X> const & rhs, S s)
                  : count_(rhs.count_)
                  , binsize_(rhs.binsize_)
                  , data_is_analyzed_(rhs.data_is_analyzed_)
                  , jacknife_bins_valid_(rhs.jacknife_bins_valid_)
                  , cannot_rebin_(rhs.cannot_rebin_)
                {
                    mean_ = slice_value(rhs.mean_, s);
                    error_ = slice_value(rhs.error_, s);
                    if (rhs.variance_opt_) 
                        variance_opt_ = slice_value(*(rhs.variance_opt_), s);
                    if (rhs.tau_opt_)
                        tau_opt_ = slice_value(*(rhs.tau_opt_), s);
                    values_.reserve(rhs.values_.size());
                    std::transform(rhs.values_.begin(), rhs.values_.end(), std::back_inserter(values_), boost::bind2nd(slice_it<X>(), s));
                    if (rhs.jacknife_bins_valid_)
                        std::transform(rhs.jack_.begin(), rhs.jack_.end(), std::back_inserter(jack_), boost::bind2nd(slice_it<X>(), s));
                }
                binned_data(AbstractSimpleObservable<value_type> const & obs) {
                    : count_(obs.count())
                    , binsize_(obs.bin_size())
                    , data_is_analyzed_(false)
                    , jacknife_bins_valid_(false)
                    , cannot_rebin_(false)
                {
                    if (count()) {
                        mean_ = obs.mean();
                        error_ = obs.error();
                        if (obs.has_variance())
                            variance_opt_ = obs.variance();
                        if (obs.has_tau())
                            tau_opt_ = obs.tau();
                        values_ = obs.bins();
                    }
                }
                inline uint64_t count() const { 
                    return count_;
                }
                inline uint64_t bin_size() const { 
                    return binsize_;
                }
                inline std::size_t bin_number() const { 
                    return values_.size(); 
                }
                inline std::vector<value_type> const & bins() const { 
                    return values_;  
                }
                inline result_type const mean() const {
                    analyze();
                    return mean_;
                }
                inline result_type const error() const {
                    analyze();
                    return error_;
                }
                inline result_type const & variance() const {
                    analyze();  
                    if (!variance_opt_)
                        boost::throw_exception(std::logic_error("observable does not have variance"));
                    return *variance_opt_;
                };
                inline time_type const & tau() const {
                    analyze();  
                    if (!tau_opt_)
                        boost::throw_exception(std::logic_error("observable does not have autocorrelation information"));
                    return *tau_opt_;
                }
                covariance_type covariance(binned_data<T> const & obs) const {
                    fill_jack();
                    obs.fill_jack();
                    if (jack_.size() && obs.jack_.size()) {
                        result_type unbiased_mean1_;
                        result_type unbiased_mean2_;
                        resize_same_as(unbiased_mean1_, jack_[0]);
                        resize_same_as(unbiased_mean2_, obs.jack_[0]);
                        if (jack_.size() != obs.jack_.size()) 
                            boost::throw_exception(std::runtime_error("unequal number of bins in calculation of covariance matrix"));
                        unbiased_mean1_ = 0;
                        unbiased_mean2_ = 0;
                        unbiased_mean1_ = std::accumulate(jack_.begin()+1, jack_.end(), unbiased_mean1_);
                        unbiased_mean2_ = std::accumulate(obs.jack_.begin()+1, obs.jack_.end(), unbiased_mean2_);
                        unbiased_mean1_ /= count_type(bin_number());
                        unbiased_mean2_ /= count_type(obs.bin_number());
                        using alps::numeric::outer_product;
                        covariance_type cov = outer_product(jack_[1],obs.jack_[1]);
                        for (uint32_t i = 1; i < bin_number(); ++i)
                            cov += outer_product(jack_[i+1],obs.jack_[i+1]);
                        cov /= count_type(bin_number());
                        cov -= outer_product(unbiased_mean1_, unbiased_mean2_);
                        cov *= count_type(bin_number() - 1);
                        return cov;
                    } else {
                        boost::throw_exception(std::runtime_error ("no binning information available for calculation of covariances"));
                        return covariance_type();
                    }
                }
                inline void set_bin_size(uint64_t binsize) {
                    collect_bins(( binsize - 1 ) / binsize_ + 1 );
                    binsize_ = binsize;
                }
                inline void set_bin_number(uint64_t bin_number) {
                    collect_bins(( values_.size() - 1 ) / bin_number + 1 );
                }
                template <class X> binned_data const & operator=(binned_data<X> const & rhs);
                template <class S> binned_data<typename element_type<T>::type> slice(S s) const {
                    return binned_data<typename element_type<T>::type>(*this,s);
                }
                // wtf? check why!!! (after 2.0b1)
                template <class X> bool operator==(binned_data<X> const & rhs) {
                    return count_ == rhs.count_
                        && binsize_ == rhs.binsize_
                        && mean_ == rhs.mean_
                        && error_ == rhs.error_
                        && variance_opt_ == rhs.variance_opt_
                        && tau_opt_ == rhs.tau_opt_
                        && values_ == rhs.values_
                    ;
                }
                binned_data<T> & operator+=(binned_data<T> const & rhs) {
                    using std::sqrt;
                    using alps::numeric::sq;
                    using alps::numeric::sqrt;
                    using boost::numeric::operators::operator+;
                    using boost::lambda::_1;
                    using boost::lambda::_2;
                    transform(rhs, _1 + _2, sqrt(sq(error_) + sq(rhs.error_)));
                }
                binned_data<T> & operator-=(binned_data<T> const & rhs) {
                    using std::sqrt;
                    using alps::numeric::sq;
                    using alps::numeric::sqrt;
                    using boost::numeric::operators::operator+;
                    using boost::numeric::operators::operator-;
                    using boost::lambda::_1;
                    using boost::lambda::_2;
                    transform(rhs, _1 - _2, sqrt(sq(error_) + sq(rhs.error_)));
                    return *this;
                }
                template <class X> binned_data<T> & operator*=(binned_data<X> const & rhs) {
                    using std::sqrt;
                    using alps::numeric::sq;
                    using alps::numeric::sqrt;
                    using boost::numeric::operators::operator+;
                    using boost::numeric::operators::operator*;
                    using boost::lambda::_1;
                    using boost::lambda::_2;
                    transform(rhs, _1 * _2, sqrt(sq(rhs.mean_) * sq(error_) + sq(mean_) * sq(rhs.error_)));
                    return *this;
                }
                template <class X> binned_data<T> & operator/=(binned_data<X> const & rhs) {
                    using std::sqrt;
                    using alps::numeric::sq;
                    using alps::numeric::sqrt;
                    using boost::numeric::operators::operator+;
                    using boost::numeric::operators::operator*;
                    using boost::numeric::operators::operator/;
                    using boost::lambda::_1;
                    using boost::lambda::_2;
                    transform(rhs, _1 / _2, sqrt(sq(rhs.mean_) * sq(error_) + sq(mean_) * sq(rhs.error_)) / sq(rhs.mean_));
                    return *this;
                }
                template <class X> binned_data<T> & operator+=(X const & rhs) {
                    using boost::numeric::operators::operator+;
                    using boost::lambda::_1;
                    transform_linear(_1 + rhs, error_, variance_opt_);
                    return *this;
                }
                template <class X> binned_data<T> & operator-=(X const & rhs) {
                    using boost::numeric::operators::operator-;
                    using boost::lambda::_1;
                    transform_linear(_1 - rhs, error_, variance_opt_);
                    return *this;
                }
                template <class X> binned_data<T> & operator*=(X const & rhs) {
                    using std::abs;
                    using alps::numeric::abs;
                    using boost::numeric::operators::operator*;
                    using boost::lambda::_1;
                    transform_linear(_1 * rhs, abs(error_ * rhs), variance_opt_ ? *variance_opt_ * rhs * rhs : boost::none_t);
                    return *this;
                }
                template <class X> binned_data<T> & operator/=(X const & rhs) {
                    using std::abs;
                    using alps::numeric::abs;
                    using boost::numeric::operators::operator/;
                    using boost::lambda::_1;
                    transform_linear(_1 / rhs, abs(error_ / rhs), variance_opt_ ? *variance_opt_ / ( rhs * rhs ) : boost::none_t);
                    return (*this);
                }
                template <class T> binned_data<T> & operator-() {
                    transform_linear(-_1);
                    return *this;
                }
                template <class X> void subtract_from(X const & x) {
                    using boost::lambda::_1;
                    using boost::numeric::operators::operator-;
                    transform_linear(x - _1, error_, variance_opt_);
                }
                template <class X> void divide(X const & x) {
                    using boost::lambda::_1;
                    error_ = x * error_ / mean_ / mean_;
                    cannot_rebin_ = true;
                    mean_ = x / mean_;
                    std::transform(values_.begin(), values_.end(), values_.begin(), ( x * bin_size() * bin_size() ) / _1);
                    fill_jack();
                    std::transform(jack_.begin(), jack_.end(), jack_.begin(), x / _1);
                }

                template <class OP> void transform_linear(OP op, value_type const & error, optional<result_type> variance_opt = boost::none_t()) {
                    if (count() == 0)
                        boost::throw_exception(std::runtime_error("the observable needs measurements"));
                    std::transform(values_.begin(), values_.end(), values_.begin(), op, error, variance_opt);
                    if (jacknife_bins_valid_)
                        std::transform(jack_.begin(), jack_.end(), jack_.begin(), op);
                }
                template <class OP> void transform(OP op, value_type const & error, optional<result_type> variance_opt = boost::none_t()) {
                    if (count() == 0)
                        boost::throw_exception(std::runtime_error("the observable needs measurements"));
                    data_is_analyzed_ = false;
                    cannot_rebin_ = true;
                    mean_ = op(mean_);
                    error_ = error;
                    variance_opt_ = variance_opt_;
                    if (!variance_opt_)
                        tau_opt_ = boost::none_t();
                    // CHANGE THE SEMANTICS OF THE BINS THROUGHOUT!
                    // bins should contain the bin mean and not bin sum!
                    // the constructor has to divide by bin size for now
                    // be careful in I/O when reading old data files that contain the sum
                    // will also fix Tama's problems
                    std::transform(values_.begin(), values_.end(), values_.begin(), op);
                    if (jacknife_bins_valid_)
                       std::transform(jack_.begin(), jack_.end(), jack_.begin(), op);
                }
                template <class X, class OP> void transform(binned_data<X> const & rhs, OP op, value_type const & error, optional<result_type> variance_opt = boost::none_t) {
                    if (count() == 0 || x.count() == 0)
                        boost::throw_exception(std::runtime_error("both observables need measurements"));
                    data_is_analyzed_= false;
                    cannot_rebin_ = true;
                    mean_ = op(mean_, rhs.mean_);
                    error_ = error;
                    variance_opt_ = variance_opt_;
                    if (!variance_opt_)
                        tau_opt_ = boost::none_t();
                    std::transform(values_.begin(), values_.end(), rhs.values_.begin(), values_.begin(), op);
                    if (rhs.jacknife_bins_valid_ && jacknife_bins_valid_)
	                    std::transform(jack_.begin(), jack_.end(), rhs.jack_.begin(), jack_.begin(), op);
                }
            private:
                void collect_bins(uint64_t howmany) {
                    if (cannot_rebin_)
                        boost::throw_exception(std::runtime_error("cannot change bins after nonlinear operations"));
                    if (values_.empty() || howmany <= 1) 
                        return;
                    uint64_t newbins = values_.size() / howmany;
                    for (uint64_t i = 0; i < newbins; ++i) {
                        values_[i] = values_[howmany * i];
                        for (uint64_t j = 1; j < howmany; ++j)
                            values_[i] += values_[howmany * i + j];
                    }
                    values_.resize(newbins);
                    binsize_ *= howmany;
                    jacknife_bins_valid_ = false;
                    data_is_analyzed_ = false;
                }
                void fill_jack() const {
                    // build jackknife data structure
                    if (bin_number() && !jacknife_bins_valid_) {
                        if (cannot_rebin_)
                          boost::throw_exception(std::runtime_error("Cannot rebuild jackknife data structure after nonlinear operations"));
                        jack_.clear();
                        jack_.resize(bin_number() + 1);
                        // Order-N initialization of jackknife data structure
                        resize_same_as(jack_[0], bin_value(0));
                        for(uint64_t j = 0; j < bin_number(); ++j) // to this point, jack_[0] = \sum_{j} x_j   (Note: x_j = (bin_value(j) / bin_size()))
                            jack_[0] += alps::numeric_cast<result_type>(values_[j]);
                        for(uint64_t i = 0; i < bin_number(); ++i) // to this point, jack_[i+1] = \sum_{j != i} x_j   (Note: x_j = (bin_value(j) / bin_size()))
                            jack_[i+1] = jack_[0] - alps::numeric_cast<result_type>(values_[i]);
                        /*  
                         *  Next, we want the following:
                         *    a)  jack_[0]   =  <x>
                         *    b)  jack_[i+1] =  <x_i>_{jacknife}
                         */
                        jack_[0] /= count_type(bin_number()); // up to this point, jack_[0] is the jacknife mean...
                        for (uint64_t j = 0; j < bin_number(); ++j)  
                            jack_[j+1] /= count_type(bin_number() - 1);
                    }
                    jacknife_bins_valid_ = true;
                }
                void analyze() const {
                    if (count() == 0)
                        boost::throw_exception(NoMeasurementsError());
                    if (data_is_analyzed_) 
                        return;
                    if (bin_number()) {
                        count_ = bin_size() * bin_number();
                        fill_jack();
                        if (jack_.size()) {
                            resize_same_as(mean_, jack_[0]);
                            resize_same_as(error_, jack_[0]);
                            result_type unbiased_mean_;
                            resize_same_as(unbiased_mean_, jack_[0]);  
                            unbiased_mean_ = 0;
                            unbiased_mean_ = std::accumulate(jack_.begin() + 1, jack_.end(), unbiased_mean_);
                            unbiased_mean_ /= count_type(bin_number());
                            mean_ = jack_[0] - (unbiased_mean_ - jack_[0]) * (count_type(bin_number() - 1));
                            error_ = 0.;
                            for (uint64_t i = 0; i < bin_number(); ++i) 
                                error_ += (jack_[i + 1] - unbiased_mean_) * (jack_[i+1] - unbiased_mean_);
                            error_ /= count_type(bin_number());
                            error_ *= count_type(bin_number() - 1);
                            error_ = std::sqrt(error_);
                        }
                        variance_opt_ = boost::none_t();
                        tau_opt_ = boost::none_t();
                    }
                    data_is_analyzed_ = true;
                }
                mutable uint64_t count_;
                mutable uint64_t binsize_;
                mutable bool data_is_analyzed_;
                mutable bool jacknife_bins_valid_;
                mutable bool cannot_rebin_;
                mutable result_type mean_;
                mutable result_type error_;
                mutable boost::optional<result_type> variance_opt_;
                mutable boost::optional<time_type> tau_opt_;
                mutable std::vector<value_type> values_;
                mutable std::vector<result_type> jack_;
        };
        #define IMPLEMENT_OPERATION(OPERATOR_NAME, OPERATOR_ASSIGN, OPERATOR_ASSIGN2)                                                                      \
            template<class T> inline binned_data<T> OPERATOR_NAME(binned_data<T> lhs, binned_data<T> const & rhs) {                                        \
                return lhs OPERATOR_ASSIGN rhs;                                                                                                            \
            }                                                                                                                                              \
            template <class T> inline binned_data<T> OPERATOR_NAME(binned_data<T> lhs, T const & rhs) {                                                    \
                return lhs OPERATOR_ASSIGN rhs;                                                                                                            \
            }                                                                                                                                              \
        template <class T> inline binned_data<std::vector<T> > OPERATOR_NAME(                                                                              \
            binned_data<std::vector<T> > lhs, typename binned_data<std::vector<T> >::element_type const & rhs_elem                                         \
        ) {                                                                                                                                                \
          std::vector<T> rhs(lhs.size(),rhs_elem);                                                                                                         \
          return lhs OPERATOR_ASSIGN rhs;                                                                                                                  \
        }                                                                                                                                                  \
        template <class T> inline binned_data<std::vector<T> > OPERATOR_NAME(                                                                              \
            typename binned_data<std::vector<T> >::element_type const & lhs_elem, binned_data<std::vector<T> > rhs                                         \
        ) {                                                                                                                                                \
          std::vector<T> lhs(rhs.size(),lhs_elem);                                                                                                         \
          return lhs OPERATOR_ASSIGN2 rhs;                                                                                                                 \
        }
        IMPLEMENT_OPERATION(operator+,+=,+)
        IMPLEMENT_OPERATION(operator-,-=,-)
        IMPLEMENT_OPERATION(operator*,*=,*)
        IMPLEMENT_OPERATION(operator/,/=,/)
        template <class T> inline binned_data<T> operator+(T const & lhs, binned_data<T> rhs) {
            return rhs += lhs;
        }
        template <class T> inline binned_data<T> operator-(T const & lhs, binned_data<T> rhs) {
            return rhs.subtract_from(lhs);
        }
        template <class T> inline binned_data<T> operator*(T const & lhs, binned_data<T> rhs) {
            return rhs *= lhs;
        }
        template <class T>  inline binned_data<T> operator/(T const & lhs, binned_data<T> const & rhs) {
            return rhs.divide(lhs);
        }
        template <class T> inline binned_data<T> pow(binned_data<T> rhs, typename binned_data<T>::element_type const & exponent) {
            if (exponent == 1.)
              return rhs;
            else {
                using std::pow;
                using std::abs;
                using alps::numeric::pow;
                using alps::numeric::abs;
                using boost::numeric::operators::operator-;
                using boost::numeric::operators::operator*;
                rhs.transform(pow(_1,_2), abs(exponent * pow(rhs.mean(), exponent - 1.) * rhs.error()));
                return rhs;
            }
        }
        template<class T> inline binned_data<T> sq(binned_data<T> rhs) {
            using alps::numeric::sq;
            using std::abs;
            using alps::numeric::abs;
            using alps::numeric::operator*;
            using boost::numeric::operators::operator*;
            rhs.transform(sq(_1), abs(2. * rhs.mean() * rhs.error()));
            return rhs;
        }
        template<class T> binned_data<T> cb(binned_data<T> rhs) {
            using alps::numeric::sq;
            using std::abs;
            using alps::numeric::abs;
            using alps::numeric::operator*;
            using boost::numeric::operators::operator*;
            rhs.transform(cb(_1), abs(3. * sq(rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<class T> binned_data<T> sqrt(binned_data<T> rhs) {
            using std::sqrt;
            using alps::numeric::sqrt;
            using std::abs;
            using alps::numeric::abs;
            using alps::numeric::operator*;
            using boost::numeric::operators::operator/;
            rhs.transform(sqrt(_1), abs(rhs.error() / (2. * sqrt(rhs.mean()))));
            return rhs;
        }
        template<class T> binned_data<T> cbrt(binned_data<T> rhs) {
            using alps::numeric::sq;
            using std::abs;
            using alps::numeric::abs;
            using std::pow;
            using alps::numeric::pow;
            using alps::numeric::operator*;
            using boost::numeric::operators::operator/;
            rhs.transform(cbrt(_1), abs(rhs.error() / (3. * sq(pow(rhs.mean(), 1. / 3)))));
            return rhs;
        }
        template<class T> binned_data<T> exp(binned_data<T> rhs) {
            using std::exp;
            using alps::numeric::exp;
            using boost::numeric::operators::operator*;
            rhs.transform(exp(_1), exp(rhs.mean()) * rhs.error());
            return rhs;
        }
        template<class T> binned_data<T> log(binned_data<T> rhs) {
            using std::log;
            using alps::numeric::log;
            using std::abs;
            using alps::numeric::abs;
            using boost::numeric::operators::operator/;
            rhs.transform(log(_1), abs(rhs.error() / rhs.mean()));
            return rhs;
        }
        template<class T> inline binned_data<T> sin(binned_data<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sin;
            using alps::numeric::sin;
            using std::cos;
            using alps::numeric::cos;
            using boost::numeric::operators::operator*;
            rhs.transform(sin(_1), abs(cos(rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<class T> binned_data<T> cos(binned_data<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sin;
            using alps::numeric::sin;
            using std::cos;
            using alps::numeric::cos;
            using boost::numeric::operators::operator-;
            using boost::numeric::operators::operator*;
            rhs.transform(cos(_1), abs(-sin(rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<class T> binned_data<T> tan(binned_data<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::tan;
            using alps::numeric::tan;
            using std::cos;
            using alps::numeric::cos;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(tan(_1), abs(1./(cos(rhs.mean()) * cos(rhs.mean())) * rhs.error()));
            return rhs;
        }
        template<class T> binned_data<T> sinh(binned_data<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sinh;
            using alps::numeric::sinh;
            using std::cosh;
            using alps::numeric::cosh;
            using boost::numeric::operators::operator*;
            rhs.transform(sinh(_1), abs(cosh(rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<class T> binned_data<T> cosh(binned_data<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sinh;
            using alps::numeric::sinh;
            using std::cosh;
            using alps::numeric::cosh;
            using boost::numeric::operators::operator*;
            rhs.transform(cosh(_1), abs(sinh(rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<class T> binned_data<T> tanh(binned_data<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::cosh;
            using alps::numeric::cosh;
            using std::tanh;
            using alps::numeric::tanh;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(tanh(_1), abs(1. / (cosh(rhs.mean()) * cosh(rhs.mean())) * rhs.error()));
            return rhs;
        }
        template<class T> binned_data<T> asin(binned_data<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sqrt;
            using alps::numeric::sqrt;
            using std::asin;
            using alps::numeric::asin;
            using alps::numeric::operator-;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(asin(_1), abs(1. / sqrt(1. - rhs.mean() * rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<class T> binned_data<T> acos(binned_data<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sqrt;
            using alps::numeric::sqrt;
            using std::acos;
            using alps::numeric::acos;
            using alps::numeric::operator-;
            using boost::numeric::operators::operator-;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(acos(_1), abs(-1. / sqrt(1. - rhs.mean() * rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<class T> binned_data<T> atan(binned_data<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::atan;
            using alps::numeric::atan;
            using alps::numeric::operator+;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(atan(_1), abs(1. / (1. + rhs.mean() * rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<class T> binned_data<T> asinh(binned_data<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sqrt;
            using alps::numeric::sqrt;
            using boost::math::asinh;
            using alps::numeric::asinh;
            using alps::numeric::operator+;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(asinh(_1), abs(1. / sqrt(rhs.mean() * rhs.mean() + 1.) * rhs.error()));
            return rhs;
        }
        template<class T> binned_data<T> acosh(binned_data<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sqrt;
            using alps::numeric::sqrt;
            using boost::math::acosh;
            using alps::numeric::acosh;
            using alps::numeric::operator-;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(acosh(_1), abs(1. / sqrt(rhs.mean() * rhs.mean() - 1.) * rhs.error()));
            return rhs;
        }
        template<class T> binned_data<T> atanh(binned_data<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using boost::math::atanh;
            using alps::numeric::atanh;
            using alps::numeric::operator-;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(atanh(_1), abs(1./(1. - rhs.mean() * rhs.mean()) * rhs.error()));
            return rhs;
        }
    }
}

#endif
