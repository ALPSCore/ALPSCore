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

/* $Id: mcdata.hpp 3545 2010-03-22 10:00:00Z gamperl $ */

#ifndef ALPS_ALEA_MCDATA_HPP
#define ALPS_ALEA_MCDATA_HPP

#include <alps/config.h>
#include <alps/alea/nan.h>
#include <alps/parser/parser.h>
#include <alps/utility/resize.hpp>
#include <alps/alea/simpleobservable.h>
#include <alps/utility/numeric_cast.hpp>
#include <alps/numeric/outer_product.hpp>
#include <alps/type_traits/average_type.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/numeric/valarray_functions.hpp>
#include <alps/type_traits/covariance_type.hpp>
#include <alps/type_traits/change_value_type.hpp>
#include <alps/numeric/vector_valarray_conversion.hpp>

#include <boost/config.hpp>
#include <boost/functional.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/optional/optional.hpp>

#include <vector>
#include <numeric>
#include <iostream>

namespace alps { 
    namespace alea {
        template <typename T> class mcdata {
            public:
                template <typename X> friend class mcdata;
                typedef T value_type;
                typedef typename change_value_type<T,double>::type time_type;
                typedef std::size_t size_type;
                typedef double count_type;
                typedef typename average_type<T>::type result_type;
                typedef typename change_value_type<T,int>::type convergence_type;
                typedef typename covariance_type<T>::type covariance_type;
                mcdata()
                    : data_is_analyzed_(true)
                    , jacknife_bins_valid_(true)
                    , cannot_rebin_(false)
                {}
                template <typename X> mcdata(mcdata<X> const & rhs)
                  : count_(rhs.count_)
                  , mean_(rhs.mean_)
                  , error_(rhs.error_)
                  , binsize_(rhs.binsize_)
                  , data_is_analyzed_(rhs.data_is_analyzed_)
                  , jacknife_bins_valid_(rhs.jacknife_bins_valid_)
                  , cannot_rebin_(rhs.cannot_rebin_)
                  , variance_opt_(rhs.variance_opt_)
                  , tau_opt_(tau_opt_)
                  , values_(rhs.values_)
                  , jack_(rhs.jack_)
                {}
                template <typename X, typename S> mcdata(mcdata<X> const & rhs, S s)
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
                mcdata(AbstractSimpleObservable<value_type> const & obs)
                    : count_(obs.count())
                    , binsize_(obs.bin_size())
                    , data_is_analyzed_(false)
                    , jacknife_bins_valid_(false)
                    , cannot_rebin_(false)
                {
                    using boost::numeric::operators::operator/;
                    if (count()) {
                        mean_ = obs.mean();
                        error_ = obs.error();
                        if (obs.has_variance())
                            variance_opt_ = obs.variance();
                        if (obs.has_tau())
                            tau_opt_ = obs.tau();
                        for (std::size_t i = 0; i < obs.bin_number(); ++i)
                            values_.push_back(obs.bin_value(i) / double(binsize_));
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
                covariance_type covariance(mcdata<T> const & obs) const {
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
                void output(std::ostream& out, boost::mpl::true_) const {
                    if(count() == 0)
                        out << "no measurements" << std::endl;
                    else {
                        out << std::setprecision(6) << alps::numeric::round<2>(mean()) << " +/- "
                            << std::setprecision(3) << alps::numeric::round<2>(error());
                        if(tau_opt_)
                            out << std::setprecision(3) <<  "; tau = " << (alps::numeric::is_nonzero<2>(error()) ? *tau_opt_ : 0);
                        out << std::setprecision(6) << std::endl;
                    }
                }
                void output_vector(std::ostream& out, boost::mpl::false_) const {
                    if(count() == 0)
                        out << "no measurements" << std::endl;
                    else
                        for (typename alps::slice_index<result_type>::type it= slices(mean_).first; it != slices(mean_).second; ++it) {
                            out << "Entry[" << slice_name(mean_, it) << "]: "
                                << alps::numeric::round<2>(slice_value(mean_, it)) << " +/- "
                                << alps::numeric::round<2>(slice_value(error_, it));
                            if(tau_opt_)
                                out << "; tau = " << (alps::numeric::is_nonzero<2>(slice_value(error_, it)) ? slice_value(*tau_opt_, it) : 0);
                            out << std::endl;
                        }
                }
                void serialize(hdf5::iarchive & ar) {
                    data_is_analyzed_ = true;
                    ar
                        >> make_pvp("count", count_)
                        >> make_pvp(ar.is_attribute("@nonlinearoperations") ? "@nonlinearoperations" : "@cannotrebin", cannot_rebin_)
                        >> make_pvp("mean/value", mean_)
                        >> make_pvp("mean/error", error_)
                    ;
                    if (ar.is_data("variance/value"))
                        ar
                            >> make_pvp("variance/value", variance_opt_)
                        ;
                    else
                        variance_opt_ = boost::none_t();
                    if (ar.is_data("tau/value"))
                        ar
                            >> make_pvp("tau/value", tau_opt_)
                        ;
                    ar
                        >> make_pvp("timeseries/data", values_)
//                        >> make_pvp("timeseries/data/@maxbinnum", max_bin_number_)
                    ;
                    if (ar.is_attribute("timeseries/data/@binsize"))
                        ar
                            >> make_pvp("timeseries/data/@binsize", binsize_)
                        ;
                    else
                        binsize_ = count_ / values_.size();
                    if (ar.is_attribute("@nonlinearoperations"))
                        values_ /= double(binsize_);
                    if ((jacknife_bins_valid_ = ar.is_data("jacknife/data")))
                        ar
                            >> make_pvp("jacknife/data", jack_)
                        ;
                }
                void serialize(hdf5::oarchive & ar) const {
                    analyze();
                    ar
                        << make_pvp("count", count_)
                        << make_pvp("@cannotrebin", cannot_rebin_)
                        << make_pvp("mean/value", mean_)
                        << make_pvp("mean/error", error_)
                    ;
                    if (variance_opt_)
                        ar
                            << make_pvp("variance/value", *variance_opt_)
                        ;
                    if (tau_opt_)
                        ar
                            << make_pvp("tau/value", *tau_opt_)
                        ;
                    ar
                        << make_pvp("timeseries/data", values_)
                        << make_pvp("timeseries/data/@binsize", binsize_)
//                        << make_pvp("timeseries/data/@maxbinnum", max_bin_number_)
                        << make_pvp("timeseries/data/@binningtype", "linear")
                    ;
                    if (jacknife_bins_valid_)
                        ar
                            << make_pvp("jacknife/data", jack_)
                            << make_pvp("jacknife/data/@binningtype", "linear")
                        ;
                }
                mcdata<T> & operator<<(mcdata<T> const & rhs) {
                    using std::sqrt;
                    using alps::numeric::sq;
                    using alps::numeric::sqrt;
                    if (rhs.count())
                        if (count()) {
                            jacknife_bins_valid_ = false;
                            data_is_analyzed_ = data_is_analyzed_ && rhs.data_is_analyzed_;
                            cannot_rebin_ = cannot_rebin_ && rhs.cannot_rebin_;
                            mean_ *= double(count_);
                            mean_ += double(rhs.count_) * rhs.mean_;
                            mean_ /= double(count_ + rhs.count_);
                            result_type tmp = error_, tmp2 = rhs.error_;
                            tmp *= error_ * sq(double(count_));
                            tmp2 *= rhs.error_ * sq(double(rhs.count_));
                            error_ = sqrt(tmp + tmp2);
                            error_ /= double(count_ + rhs.count_);
                            if(variance_opt_) {
                                *variance_opt_ *= double(count_);
                                *variance_opt_ += double(rhs.count_) * *rhs.variance_opt_;
                                *variance_opt_ /= double(count_ + rhs.count_);
                            } else
                                variance_opt_ = boost::none_t();
                            if(tau_opt_ && rhs.tau_opt_) {
                                *tau_opt_ *= double(count_);
                                *tau_opt_ += double(rhs.count_) * *rhs.tau_opt_;
                                *tau_opt_ /= double(count_ + rhs.count_);
                            } else
                                tau_opt_ = boost::none_t();
                            count_ += rhs.count();
                            if (binsize_ <= rhs.bin_size()) {
                                if (binsize_ < rhs.bin_size())
                                    set_bin_size(rhs.bin_size());
                                std::copy(rhs.values_.begin(), rhs.values_.end(), std::back_inserter(values_));
                            } else {
                                mcdata<T> tmp(rhs);
                                tmp.set_bin_size(binsize_);
                                std::copy(tmp.values_.begin(), tmp.values_.end(), std::back_inserter(values_));
                            }
//                            if (max_bin_number_ && max_bin_number_ < bin_number())
//                              set_bin_number(max_bin_number_);
                            } else
                                *this = rhs;
                    return *this;
                }
                mcdata<T> & operator=(mcdata<T> rhs) {
                // TODO: fixit
//                    rhs.swap(*this);
                    return *this;
                }
                template <typename S> mcdata<typename element_type<T>::type> slice(S s) const {
                    return mcdata<typename element_type<T>::type>(*this,s);
                }
                // wtf? check why!!! (after 2.0b1)
                template <typename X> bool operator==(mcdata<X> const & rhs) {
                    return count_ == rhs.count_
                        && binsize_ == rhs.binsize_
                        && mean_ == rhs.mean_
                        && error_ == rhs.error_
                        && variance_opt_ == rhs.variance_opt_
                        && tau_opt_ == rhs.tau_opt_
                        && values_ == rhs.values_
                    ;
                }
                mcdata<T> & operator+=(mcdata<T> const & rhs) {
                    using std::sqrt;
                    using alps::numeric::sq;
                    using alps::numeric::sqrt;
                    using boost::numeric::operators::operator+;
                    using boost::lambda::_1;
                    using boost::lambda::_2;
                    transform(rhs, _1 + _2, sqrt(sq(error_) + sq(rhs.error_)));
                }
                mcdata<T> & operator-=(mcdata<T> const & rhs) {
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
                template <typename X> mcdata<T> & operator*=(mcdata<X> const & rhs) {
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
                template <typename X> mcdata<T> & operator/=(mcdata<X> const & rhs) {
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
                template <typename X> mcdata<T> & operator+=(X const & rhs) {
                    using boost::numeric::operators::operator+;
                    using boost::lambda::_1;
                    transform_linear(_1 + rhs, error_, variance_opt_);
                    return *this;
                }
                template <typename X> mcdata<T> & operator-=(X const & rhs) {
                    using boost::numeric::operators::operator-;
                    using boost::lambda::_1;
                    transform_linear(_1 - rhs, error_, variance_opt_);
                    return *this;
                }
                template <typename X> mcdata<T> & operator*=(X const & rhs) {
                    using std::abs;
                    using alps::numeric::abs;
                    using boost::numeric::operators::operator*;
                    using boost::lambda::_1;
                    transform_linear(_1 * rhs, abs(error_ * rhs), variance_opt_ ? *variance_opt_ * rhs * rhs : boost::none_t());
                    return *this;
                }
                template <typename X> mcdata<T> & operator/=(X const & rhs) {
                    using std::abs;
                    using alps::numeric::abs;
                    using boost::numeric::operators::operator/;
                    using boost::lambda::_1;
                    transform_linear(_1 / rhs, abs(error_ / rhs), variance_opt_ ? *variance_opt_ / ( rhs * rhs ) : boost::none_t());
                    return (*this);
                }
                mcdata<T> & operator-() {
                    using boost::lambda::_1;
                    transform_linear(-_1);
                    return *this;
                }
                template <typename X> void subtract_from(X const & x) {
                    using boost::lambda::_1;
                    using boost::numeric::operators::operator-;
                    transform_linear(x - _1, error_, variance_opt_);
                }
                template <typename X> void divide(X const & x) {
                    using boost::lambda::_1;
                    error_ = x * error_ / mean_ / mean_;
                    cannot_rebin_ = true;
                    mean_ = x / mean_;
                    std::transform(values_.begin(), values_.end(), values_.begin(), ( x * bin_size() * bin_size() ) / _1);
                    fill_jack();
                    std::transform(jack_.begin(), jack_.end(), jack_.begin(), x / _1);
                }
                template <typename OP> void transform_linear(OP op, value_type const & error, boost::optional<result_type> variance_opt = boost::none_t()) {
                    if (count() == 0)
                        boost::throw_exception(std::runtime_error("the observable needs measurements"));
                    std::transform(values_.begin(), values_.end(), values_.begin(), op, error, variance_opt);
                    if (jacknife_bins_valid_)
                        std::transform(jack_.begin(), jack_.end(), jack_.begin(), op);
                }
                template <typename OP> void transform(OP op, value_type const & error, boost::optional<result_type> variance_opt = boost::none_t()) {
                    if (count() == 0)
                        boost::throw_exception(std::runtime_error("the observable needs measurements"));
                    data_is_analyzed_ = false;
                    cannot_rebin_ = true;
                    mean_ = op(mean_);
                    error_ = error;
                    variance_opt_ = variance_opt_;
                    if (!variance_opt_)
                        tau_opt_ = boost::none_t();
                    std::transform(values_.begin(), values_.end(), values_.begin(), op);
                    if (jacknife_bins_valid_)
                       std::transform(jack_.begin(), jack_.end(), jack_.begin(), op);
                }
                template <typename X, typename OP> void transform(mcdata<X> const & rhs, OP op, value_type const & error, boost::optional<result_type> variance_opt = boost::none_t()) {
                    if (count() == 0 || rhs.count() == 0)
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
                        resize_same_as(jack_[0], values_[0]);
                        for(uint64_t j = 0; j < bin_number(); ++j) // to this point, jack_[0] = \sum_{j} x_j   (Note: x_j = (values_[j] / bin_size()))
                            jack_[0] += alps::numeric_cast<result_type>(values_[j] * binsize_);
                        for(uint64_t i = 0; i < bin_number(); ++i) // to this point, jack_[i+1] = \sum_{j != i} x_j   (Note: x_j = (values_[j] / bin_size()))
                            jack_[i+1] = jack_[0] - alps::numeric_cast<result_type>(values_[i] * binsize_);
                        //  Next, we want the following:
                        //    a)  jack_[0]   =  <x>
                        //    b)  jack_[i+1] =  <x_i>_{jacknife}
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
        template <typename T> inline std::ostream& operator<<(std::ostream & out, mcdata<T> const & obs) {
            obs.output(out, typename boost::is_scalar<T>::type());
            return out;
        }
        #define ALPS_ALEA_MCDATA_IMPLEMENT_OPERATION(OPERATOR_NAME, OPERATOR_ASSIGN, OPERATOR_ASSIGN2)                                                                      \
            template<typename T> inline mcdata<T> OPERATOR_NAME(mcdata<T> lhs, mcdata<T> const & rhs) {                                                    \
                return lhs OPERATOR_ASSIGN rhs;                                                                                                            \
            }                                                                                                                                              \
            template <typename T> inline mcdata<T> OPERATOR_NAME(mcdata<T> lhs, T const & rhs) {                                                           \
                return lhs OPERATOR_ASSIGN rhs;                                                                                                            \
            }                                                                                                                                              \
            template <typename T> inline mcdata<std::vector<T> > OPERATOR_NAME(mcdata<std::vector<T> > lhs, std::vector<T> const & rhs_elem) {             \
                std::vector<T> rhs(lhs.size(),rhs_elem);                                                                                                   \
                return lhs OPERATOR_ASSIGN rhs;                                                                                                            \
            }                                                                                                                                              \
            template <typename T> inline mcdata<std::vector<T> > OPERATOR_NAME(std::vector<T> const & lhs_elem, mcdata<std::vector<T> > rhs) {             \
                std::vector<T> lhs(rhs.size(),lhs_elem);                                                                                                   \
                return lhs OPERATOR_ASSIGN2 rhs;                                                                                                           \
            }
        ALPS_ALEA_MCDATA_IMPLEMENT_OPERATION(operator+,+=,+)
        ALPS_ALEA_MCDATA_IMPLEMENT_OPERATION(operator-,-=,-)
        ALPS_ALEA_MCDATA_IMPLEMENT_OPERATION(operator*,*=,*)
        ALPS_ALEA_MCDATA_IMPLEMENT_OPERATION(operator/,/=,/)
        #undef ALPS_ALEA_MCDATA_IMPLEMENT_OPERATION
        template <typename T> inline mcdata<T> operator+(T const & lhs, mcdata<T> rhs) {
            return rhs += lhs;
        }
        template <typename T> inline mcdata<T> operator-(T const & lhs, mcdata<T> rhs) {
            rhs.subtract_from(lhs);
            return rhs;
        }
        template <typename T> inline mcdata<T> operator*(T const & lhs, mcdata<T> rhs) {
            return rhs *= lhs;
        }
        template <typename T>  inline mcdata<T> operator/(T const & lhs, mcdata<T> rhs) {
            rhs.divide(lhs);
            return rhs;
        }
        template <typename T> mcdata<T> pow(mcdata<T> rhs, double exponent) {
            if (exponent == 1.)
              return rhs;
            else {
                using std::pow;
                using std::abs;
                using alps::numeric::pow;
                using alps::numeric::abs;
                using boost::lambda::_1;
                using boost::lambda::bind;
                using boost::numeric::operators::operator-;
                using boost::numeric::operators::operator*;
                rhs.transform(bind<T>(pow<typename mcdata<T>::value_type, double>, _1, exponent), abs(exponent * pow(rhs.mean(), exponent - 1.) * rhs.error()));
                return rhs;
            }
        }
        template<typename T> inline mcdata<T> sq(mcdata<T> rhs) {
            using alps::numeric::sq;
            using std::abs;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::abs;
            using alps::numeric::operator*;
            using boost::numeric::operators::operator*;
            rhs.transform(bind<T>(sq<typename mcdata<T>::value_type>, _1), abs(2. * rhs.mean() * rhs.error()));
            return rhs;
        }
        template<typename T> mcdata<T> cb(mcdata<T> rhs) {
            using alps::numeric::sq;
            using std::abs;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::abs;
            using alps::numeric::operator*;
            using boost::numeric::operators::operator*;
            rhs.transform(bind<T>(cb<typename mcdata<T>::value_type>, _1), abs(3. * sq(rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<typename T> mcdata<T> sqrt(mcdata<T> rhs) {
            using std::sqrt;
            using alps::numeric::sqrt;
            using std::abs;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::abs;
            using alps::numeric::operator*;
            using boost::numeric::operators::operator/;
            rhs.transform(bind<T>(sqrt<typename mcdata<T>::value_type>, _1), abs(rhs.error() / (2. * sqrt(rhs.mean()))));
            return rhs;
        }
        template<typename T> mcdata<T> cbrt(mcdata<T> rhs) {
            using alps::numeric::sq;
            using std::abs;
            using alps::numeric::abs;
            using std::pow;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::pow;
            using alps::numeric::operator*;
            using boost::numeric::operators::operator/;
            rhs.transform(bind<T>(cbrt<typename mcdata<T>::value_type>, _1), abs(rhs.error() / (3. * sq(pow(rhs.mean(), 1. / 3)))));
            return rhs;
        }
        template<typename T> mcdata<T> exp(mcdata<T> rhs) {
            using std::exp;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::exp;
            using boost::numeric::operators::operator*;
            rhs.transform(bind<T>(exp<typename mcdata<T>::value_type>, _1), exp(rhs.mean()) * rhs.error());
            return rhs;
        }
        template<typename T> mcdata<T> log(mcdata<T> rhs) {
            using std::log;
            using alps::numeric::log;
            using std::abs;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::abs;
            using boost::numeric::operators::operator/;
            rhs.transform(bind<T>(log<typename mcdata<T>::value_type>, _1), abs(rhs.error() / rhs.mean()));
            return rhs;
        }
        template<typename T> inline mcdata<T> sin(mcdata<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sin;
            using alps::numeric::sin;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using std::cos;
            using alps::numeric::cos;
            using boost::numeric::operators::operator*;
            rhs.transform(bind<T>(sin<typename mcdata<T>::value_type>, _1), abs(cos(rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<typename T> mcdata<T> cos(mcdata<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sin;
            using alps::numeric::sin;
            using std::cos;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::cos;
            using boost::numeric::operators::operator-;
            using boost::numeric::operators::operator*;
            rhs.transform(bind<T>(cos<typename mcdata<T>::value_type>, _1), abs(-sin(rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<typename T> mcdata<T> tan(mcdata<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::tan;
            using alps::numeric::tan;
            using std::cos;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::cos;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(bind<T>(tan<typename mcdata<T>::value_type>, _1), abs(1./(cos(rhs.mean()) * cos(rhs.mean())) * rhs.error()));
            return rhs;
        }
        template<typename T> mcdata<T> sinh(mcdata<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sinh;
            using alps::numeric::sinh;
            using std::cosh;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::cosh;
            using boost::numeric::operators::operator*;
            rhs.transform(bind<T>(sinh<typename mcdata<T>::value_type>, _1), abs(cosh(rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<typename T> mcdata<T> cosh(mcdata<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sinh;
            using alps::numeric::sinh;
            using std::cosh;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::cosh;
            using boost::numeric::operators::operator*;
            rhs.transform(bind<T>(cosh<typename mcdata<T>::value_type>, _1), abs(sinh(rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<typename T> mcdata<T> tanh(mcdata<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::cosh;
            using alps::numeric::cosh;
            using std::tanh;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::tanh;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(bind<T>(tanh<typename mcdata<T>::value_type>, _1), abs(1. / (cosh(rhs.mean()) * cosh(rhs.mean())) * rhs.error()));
            return rhs;
        }
        template<typename T> mcdata<T> asin(mcdata<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sqrt;
            using alps::numeric::sqrt;
            using std::asin;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::asin;
            using alps::numeric::operator-;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(bind<T>(asin<typename mcdata<T>::value_type>, _1), abs(1. / sqrt(1. - rhs.mean() * rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<typename T> mcdata<T> acos(mcdata<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sqrt;
            using alps::numeric::sqrt;
            using std::acos;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::acos;
            using alps::numeric::operator-;
            using boost::numeric::operators::operator-;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(bind<T>(acos<typename mcdata<T>::value_type>, _1), abs(-1. / sqrt(1. - rhs.mean() * rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<typename T> mcdata<T> atan(mcdata<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::atan;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::atan;
            using alps::numeric::operator+;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(bind<T>(atan<typename mcdata<T>::value_type>, _1), abs(1. / (1. + rhs.mean() * rhs.mean()) * rhs.error()));
            return rhs;
        }
        template<typename T> mcdata<T> asinh(mcdata<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sqrt;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::sqrt;
            using boost::math::asinh;
            using alps::numeric::asinh;
            using alps::numeric::operator+;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(bind<T>(asinh<typename mcdata<T>::value_type>, _1), abs(1. / sqrt(rhs.mean() * rhs.mean() + 1.) * rhs.error()));
            return rhs;
        }
        template<typename T> mcdata<T> acosh(mcdata<T> rhs) {
            using std::abs;
            using alps::numeric::abs;
            using std::sqrt;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::sqrt;
            using boost::math::acosh;
            using alps::numeric::acosh;
            using alps::numeric::operator-;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(bind<T>(acosh<typename mcdata<T>::value_type>, _1), abs(1. / sqrt(rhs.mean() * rhs.mean() - 1.) * rhs.error()));
            return rhs;
        }
        template<typename T> mcdata<T> atanh(mcdata<T> rhs) {
            using std::abs;
            using boost::lambda::_1;
            using boost::lambda::bind;
            using alps::numeric::abs;
            using boost::math::atanh;
            using alps::numeric::atanh;
            using alps::numeric::operator-;
            using boost::numeric::operators::operator*;
            using alps::numeric::operator/;
            rhs.transform(bind<T>(atanh<typename mcdata<T>::value_type>, _1), abs(1./(1. - rhs.mean() * rhs.mean()) * rhs.error()));
            return rhs;
        }
    }
}
#endif
