/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2008 by Matthias Troyer <troyer@comp-phys.org>,
*                            Beat Ammon <beat.ammon@bluewin.ch>,
*                            Andreas Laeuchli <laeuchli@comp-phys.org>,
*                            Synge Todo <wistaria@comp-phys.org>,
*                            Andreas Lange <alange@phys.ethz.ch>
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

/* $Id$ */

#ifndef ALPS_ALEA_SIMPLEOBSDATA_H
#define ALPS_ALEA_SIMPLEOBSDATA_H

#include <alps/config.h>
#include <alps/alea/nan.h>
#include <alps/alea/simpleobservable.h>
#include <alps/parser/parser.h>

#include <boost/lambda/lambda.hpp>
#include <boost/functional.hpp>
#include <iostream>
#include <numeric>
#include <vector>
#ifdef ALPS_HAVE_VALARRAY
# include <valarray>
#endif

#ifndef ALPS_WITHOUT_OSIRIS
# include <alps/osiris/std/valarray.h>
#endif

#ifdef ALPS_HAVE_VALARRAY
template <class T> std::ostream& operator<<(std::ostream& o, const std::valarray<T>&) { return o;}
#endif

namespace alps {

class RealObsevaluatorXMLHandler;
class RealVectorObsevaluatorXMLHandler;

using namespace boost::lambda;

//=======================================================================
// SimpleObservableData
//
// Observable class with variable autocorrelation analysis and binning
//-----------------------------------------------------------------------


template <class T>
class SimpleObservableData
{
public:
  template <class X>
  friend class SimpleObservableData;

  friend class RealObsevaluatorXMLHandler;
  friend class RealVectorObsevaluatorXMLHandler;

  typedef T value_type;
  typedef typename obs_value_traits<T>::time_type time_type;
  typedef typename obs_value_traits<T>::size_type size_type;
  typedef typename obs_value_traits<T>::count_type count_type;
  typedef typename obs_value_traits<T>::result_type result_type;
  typedef typename obs_value_traits<T>::convergence_type convergence_type;
  typedef typename obs_value_traits<T>::label_type label_type;
  typedef typename obs_value_traits<T>::covariance_type covariance_type;

  // constructors
  SimpleObservableData();
  template <class U, class S>
  SimpleObservableData(const SimpleObservableData<U>& x, S s);
  SimpleObservableData(const AbstractSimpleObservable<value_type>& obs);
  SimpleObservableData(std::istream&, const XMLTag&, label_type& );

  template <class S>
  SimpleObservableData<typename obs_value_slice<T,S>::value_type> slice(S s) {
    return SimpleObservableData<typename obs_value_slice<T,S>::value_type>(*this, s);
  }

  SimpleObservableData const& operator=(const SimpleObservableData& x);

  void read_xml(std::istream&, const XMLTag&, label_type& label);
  void read_xml_scalar(std::istream&, const XMLTag&);
  void read_xml_vector(std::istream&, const XMLTag&, label_type& label);
  
  inline ALPS_DUMMY_VOID set_thermalization(uint32_t todiscard);
  inline uint32_t get_thermalization() const;
  inline bool can_set_thermalization() const { return can_set_thermal_ && !nonlinear_operations_;}

  uint64_t count() const { return changed_ ? (bin_size()*bin_number() == 0 ? count_ : bin_size()*bin_number()) : count_;}
  inline const result_type& mean() const;
  inline const result_type& error() const;
  inline const convergence_type& converged_errors() const;
  inline const convergence_type& any_converged_errors() const;
  inline const result_type& variance() const;
  inline const time_type& tau() const;
  //inline const value_type& min() const;
  //inline const value_type& max() const;
  
  covariance_type covariance(const SimpleObservableData<T>) const;

  bool has_variance() const { return has_variance_;}
  bool has_tau() const { return has_tau_;}
  //bool has_minmax() const { return has_minmax_;}
  bool has_minmax() const { return false;} //this interface is disabled - keeping track of min and max is too expensive

  uint64_t bin_size() const { return binsize_;}
  uint64_t bin_number() const { return values_.size()-discardedbins_;}
  uint64_t bin_number2() const { return discardedbins_ ? 0 : values2_.size();}
  const value_type& bin_value(uint64_t i) const {
    return values_[i+discardedbins_];
  }
  const value_type& bin_value2(uint64_t i) const {
    return values2_[i+discardedbins_];
  }
  
  template <class S>
    SimpleObservableData<typename obs_value_slice<T,S>::value_type> slice(S s) const
      {
        return SimpleObservableData<typename obs_value_slice<T,S>::value_type>(*this,s);
      }

  ALPS_DUMMY_VOID compact();
  
#ifndef ALPS_WITHOUT_OSIRIS
  void extract_timeseries(ODump& dump) const;
  void save(ODump& dump) const;
  void load(IDump& dump);
#endif
 
  inline void set_bin_size(uint64_t);
  inline void set_bin_number(uint64_t);
 
  // collect information from many data objects
  void collect_from(const std::vector<SimpleObservableData<T> >& runs);

  // unary operation: neagtion
  void negate();
  
  // operations with constant
  template <class X> SimpleObservableData<T>& operator+=(X);
  template <class X> SimpleObservableData<T>& operator-=(X);
  template <class X> SimpleObservableData<T>& operator*=(X);
  template <class X> SimpleObservableData<T>& operator/=(X);
  template<class X> void subtract_from(const X& x);
  template<class X> void divide(const X& x);
  

  // operations with another observable
  SimpleObservableData<T>& operator+=(const SimpleObservableData<T>&);
  SimpleObservableData<T>& operator-=(const SimpleObservableData<T>&);
  template <class X>
  SimpleObservableData<T>& operator*=(const SimpleObservableData<X>&);
  template <class X>
  SimpleObservableData<T>& operator/=(const SimpleObservableData<X>&);

  template <class OP> void transform(OP op);

  std::string evaluation_method(Target t) const;

protected:
  void collect_bins(uint64_t howmany);
  void analyze() const;
  void jackknife() const;
  void fill_jack() const;

  template <class X, class OP>
  void transform(const SimpleObservableData<X>& x, OP op, double factor=1.);
  template <class OP> void transform_linear(OP op);

private:  
  mutable uint64_t count_;          

  mutable bool has_variance_;
  mutable bool has_tau_;
  mutable bool has_minmax_;
  mutable bool can_set_thermal_;

  mutable uint64_t binsize_;
  mutable uint32_t thermalcount_; 
  mutable uint32_t discardedmeas_;
  mutable uint32_t discardedbins_;
    
  bool changed_;
  mutable bool valid_;
  mutable bool jack_valid_;
  bool nonlinear_operations_; // nontrivial operations
    
  mutable result_type mean_;     // valid only if (valid_)
  mutable result_type error_;    // valid only if (valid_)
  mutable result_type variance_; // valid only if (valid_ && has_variance_)
  mutable time_type tau_;        // valid only if (valid_ && has_tau_)
  //mutable value_type min_, max_; // valid only if (valid_ && has_minmax_)
  
  mutable std::vector<value_type> values_;
  mutable std::vector<value_type> values2_;
  mutable std::vector<result_type> jack_;
  
  mutable convergence_type converged_errors_;
  mutable convergence_type any_converged_errors_;
  std::string eval_method_;
};

  
template <class T>
SimpleObservableData<T>::SimpleObservableData()
 : count_(0),
   has_variance_(false),
   has_tau_(false),
   has_minmax_(false),
   can_set_thermal_(false),
   binsize_(0),
   thermalcount_(0),
   discardedmeas_(0),
   discardedbins_(0),
   changed_(false),
   valid_(true),
   jack_valid_(true),
   nonlinear_operations_(false),
   mean_(), error_(), variance_(), tau_(), //min_(), max_(),
   values_(), values2_(), jack_(), converged_errors_(), any_converged_errors_()
{}

template <class T>
template <class U, class S>
inline
SimpleObservableData<T>::SimpleObservableData(const SimpleObservableData<U>& x, S s)
 : count_(x.count_),          
   has_variance_(x.has_variance_),
   has_tau_(x.has_tau_),
   has_minmax_(x.has_minmax_),
   can_set_thermal_(x.can_set_thermal_),
   binsize_(x.binsize_),
   thermalcount_(x.thermalcount_),
   discardedmeas_(x.discardedmeas_),
   discardedbins_(x.discardedbins_),
   changed_(x.changed_),
   valid_(x.valid_),
   jack_valid_(x.jack_valid_),
   nonlinear_operations_(x.nonlinear_operations_),
   mean_(obs_value_slice<typename obs_value_traits<U>::result_type,S>()(x.mean_, s)),
   error_(obs_value_slice<typename obs_value_traits<U>::result_type,S>()(x.error_, s)),
   variance_(has_variance_ ? obs_value_slice<typename obs_value_traits<U>::result_type,S>()(x.variance_, s) : result_type()),
   tau_(has_tau_ ? obs_value_slice<typename obs_value_traits<U>::time_type,S>()(x.tau_, s) : time_type()),
   //min_(has_minmax_ ? obs_value_slice<U,S>()(x.min_, s) : result_type()), 
   //max_(has_minmax_ ? obs_value_slice<U,S>()(x.max_, s) : result_type()),
   values_(x.values_.size()), 
   values2_(x.values2_.size()), 
   jack_(x.jack_.size()),
   converged_errors_(obs_value_slice<typename obs_value_traits<U>::convergence_type,S>()(x.converged_errors_,s)),
   any_converged_errors_(obs_value_slice<typename obs_value_traits<U>::convergence_type,S>()(x.any_converged_errors_,s))
{
  values_.resize(x.values_.size());
  std::transform(x.values_.begin(), x.values_.end(), values_.begin(),
                 boost::bind2nd(obs_value_slice<U,S>(),s));
  values2_.resize(x.values2_.size());
  std::transform(x.values2_.begin(), x.values2_.end(), values2_.begin(),
                 boost::bind2nd(obs_value_slice<U,S>(),s));
  if (jack_valid_) {
    jack_.resize(x.jack_.size());
    std::transform(x.jack_.begin(), x.jack_.end(), jack_.begin(),
                   boost::bind2nd(obs_value_slice<U,S>(),s));
  }
}


template <class T>
inline
SimpleObservableData<T> const& SimpleObservableData<T>::operator=(const SimpleObservableData<T>& x)
 {
   count_=x.count_;        
   has_variance_=x.has_variance_;
   has_tau_=x.has_tau_;
   has_minmax_=false; //x.has_minmax_;
   can_set_thermal_=x.can_set_thermal_;
   binsize_=x.binsize_;
   thermalcount_=x.thermalcount_;
   discardedmeas_=x.discardedmeas_;
   discardedbins_=x.discardedbins_;
   changed_=x.changed_;
   valid_=x.valid_;
   jack_valid_=x.jack_valid_;
   nonlinear_operations_=x.nonlinear_operations_;
   obs_value_traits<result_type>::copy(mean_,x.mean_);
   obs_value_traits<result_type>::copy(error_,x.error_);
   obs_value_traits<result_type>::copy(variance_,x.variance_);
   obs_value_traits<time_type>::copy(tau_,x.tau_);
   //obs_value_traits<value_type>::copy(min_,x.min_);
   //obs_value_traits<value_type>::copy(max_,x.max_);
   values_=x.values_; 
   values2_=x.values2_; 
   jack_=x.jack_;
   
   obs_value_traits<convergence_type>::copy(converged_errors_,x.converged_errors_);
   obs_value_traits<convergence_type>::copy(any_converged_errors_,x.any_converged_errors_);
   
  return *this;
}


template <class T>
SimpleObservableData<T>::SimpleObservableData(const AbstractSimpleObservable<T>& obs)
 : count_(obs.count()),
   has_variance_(obs.has_variance()),
   has_tau_(obs.has_tau()),
   has_minmax_(false/*obs.has_minmax()*/),
   can_set_thermal_(obs.can_set_thermalization()),
   binsize_(obs.bin_size()),
   thermalcount_(obs.get_thermalization()),
   discardedmeas_(0),
   discardedbins_(0),
   changed_(false),
   valid_(false),
   jack_valid_(false),
   nonlinear_operations_(false),
   mean_(), error_(), variance_(), tau_(),// min_(), max_(),
   values_(), values2_(), jack_()
{
  if (count()) {
    obs_value_traits<result_type>::copy(mean_, obs.mean());
    obs_value_traits<result_type>::copy(error_, obs.error());
    if (has_variance())
      obs_value_traits<result_type>::copy(variance_, obs.variance());
    if (has_tau())
      obs_value_traits<time_type>::copy(tau_, obs.tau());
    /*if (has_minmax()) {
      obs_value_traits<result_type>::copy(min_, obs.min());
      obs_value_traits<result_type>::copy(max_, obs.max());
    }*/

    for (uint64_t i = 0; i < obs.bin_number(); ++i)
      values_.push_back(obs.bin_value(i));
    for (uint64_t i = 0; i < obs.bin_number2(); ++i)
      values2_.push_back(obs.bin_value2(i));
    obs_value_traits<convergence_type>::copy(converged_errors_, obs.converged_errors());
    obs_value_traits<convergence_type>::copy(any_converged_errors_, obs.converged_errors());
    
    if (bin_size() != 1 && bin_number() > 128) set_bin_number(128);
  }
}

template <class T>
SimpleObservableData<T>::SimpleObservableData(std::istream& infile, const XMLTag& intag, label_type& l)
  : count_(0),
    has_variance_(false),
    has_tau_(false),
    has_minmax_(false),
    can_set_thermal_(false),
    binsize_(0),
    thermalcount_(0),
    discardedmeas_(0),
    discardedbins_(0),
    changed_(false),
    valid_(true),
    jack_valid_(false),
    nonlinear_operations_(false),
    mean_(), error_(), variance_(), tau_(), //min_(), max_(),
    values_(), values2_(), jack_()
{
  read_xml(infile,intag,l);
}

inline double text_to_double(const std::string& val) 
{
  return ((val=="NaN" || val=="nan" || val=="NaNQ") ? alps::nan() :
          ((val=="INF" || val=="Inf" || val == "inf") ? alps::inf() :
           ((val=="-INF" || val=="-Inf" || val == "-inf") ? alps::ninf() :
            boost::lexical_cast<double, std::string>(val))));
}

template <class T>
void SimpleObservableData<T>::read_xml_scalar(std::istream& infile, const XMLTag& intag)
{
  if (intag.name != "SCALAR_AVERAGE")
    boost::throw_exception(std::runtime_error ("Encountered tag <" +intag.name +
 "> instead of <SCALAR_AVERAGE>"));
  if (intag.type ==XMLTag::SINGLE)
    return;

  XMLTag tag = parse_tag(infile);
  while (tag.name !="/SCALAR_AVERAGE") {
    if (tag.name=="COUNT") {
      if (tag.type !=XMLTag::SINGLE) {
        count_ = boost::lexical_cast<uint64_t,std::string>(parse_content(infile));
        check_tag(infile,"/COUNT");
      }
    }
    else if (tag.name=="MEAN") {
      if (tag.type !=XMLTag::SINGLE) {
        mean_=text_to_double(parse_content(infile));
        check_tag(infile,"/MEAN");
      }
    }
    else if (tag.name=="ERROR") {
      if (tag.type !=XMLTag::SINGLE) {
        error_=text_to_double(parse_content(infile));
        eval_method_=tag.attributes["method"]; 
        converged_errors_=(tag.attributes["converged"]=="no" ? NOT_CONVERGED : 
                           tag.attributes["converged"]=="maybe" ? MAYBE_CONVERGED : CONVERGED);
        any_converged_errors_ = converged_errors_;
        check_tag(infile,"/ERROR");
      }
    }
    else if (tag.name=="VARIANCE") {
      if (tag.type !=XMLTag::SINGLE) {
        has_variance_=true;
        variance_=text_to_double(parse_content(infile));
        check_tag(infile,"/VARIANCE");
      }
    }
    else if (tag.name=="AUTOCORR") {
      if (tag.type !=XMLTag::SINGLE) {
        has_tau_=true;
        tau_=text_to_double(parse_content(infile));
        check_tag(infile,"/AUTOCORR");
      }
    }
    else 
      skip_element(infile,tag);
    tag = parse_tag(infile);
  }
}

template <class T>
void SimpleObservableData<T>::read_xml_vector(std::istream& infile, const XMLTag& intag, label_type& label)
{
  if (intag.name != "VECTOR_AVERAGE")
    boost::throw_exception(std::runtime_error ("Encountered tag <" + intag.name + "> instead of <VECTOR_AVERAGE>"));
  if (intag.type == XMLTag::SINGLE)
    return;
  XMLTag tag(intag);
  std::size_t s = boost::lexical_cast<std::size_t,std::string>(tag.attributes["nvalues"]);
  obs_value_traits<result_type>::resize(mean_,s);
  obs_value_traits<result_type>::resize(error_,s);
  obs_value_traits<result_type>::resize(variance_,s);
  obs_value_traits<time_type>::resize(tau_,s);
  obs_value_traits<convergence_type>::resize(converged_errors_,s);
  obs_value_traits<convergence_type>::resize(any_converged_errors_,s);
  obs_value_traits<label_type>::resize(label,s);
  
  tag = parse_tag(infile);
  int i=0;
  while (tag.name =="SCALAR_AVERAGE") {
    label[i]=tag.attributes["indexvalue"];
    tag = parse_tag(infile);
    while (tag.name !="/SCALAR_AVERAGE") {
      if (tag.name=="COUNT") {
        if (tag.type != XMLTag::SINGLE) {
          count_=boost::lexical_cast<uint64_t,std::string>(parse_content(infile));
          check_tag(infile,"/COUNT");
        }
      }
      else if (tag.name=="MEAN") {
        if (tag.type !=XMLTag::SINGLE) {
          mean_[i]=text_to_double(parse_content(infile));
          check_tag(infile,"/MEAN");
        }
      }
      else if (tag.name=="ERROR") {
        if (tag.type != XMLTag::SINGLE) {
          error_[i]=text_to_double(parse_content(infile));
          converged_errors_[i] =(tag.attributes["converged"]=="no" ? NOT_CONVERGED : 
                                 tag.attributes["converged"]=="maybe" ? MAYBE_CONVERGED : CONVERGED);
          any_converged_errors_[i] = converged_errors_[i];
          eval_method_=tag.attributes["method"]; 
          check_tag(infile,"/ERROR");
        }
      }
      else if (tag.name=="VARIANCE") {
        if (tag.type !=XMLTag::SINGLE) {
          has_variance_=true;
          variance_[i]=text_to_double(parse_content(infile));
          check_tag(infile,"/VARIANCE");
        }
      }
      else if (tag.name=="AUTOCORR") {
        if (tag.type !=XMLTag::SINGLE) {
          has_tau_=true;
          tau_[i]=text_to_double(parse_content(infile));
          check_tag(infile,"/AUTOCORR");
        }
      }
      else 
        skip_element(infile,tag);
      tag = parse_tag(infile);
    }
    ++i;
    tag = parse_tag(infile);
  }
  if (tag.name!="/VECTOR_AVERAGE")
    boost::throw_exception(std::runtime_error("Encountered unknow tag <"+tag.name+"> in <VECTOR_AVERAGE>"));
}

namespace detail {

template <bool arrayvalued> struct input_helper {};
  
template <> struct input_helper<true>
{
  template <class T, class L>
  static void read_xml(SimpleObservableData<T>& obs, std::istream& infile, const XMLTag& tag, L& l) {
    obs.read_xml_vector(infile,tag,l);
  }
};
  
template <> struct input_helper<false>
{
  template <class T, class L>
  static void read_xml(SimpleObservableData<T>& obs, std::istream& infile, const XMLTag& tag,L&) {
    obs.read_xml_scalar(infile,tag);
  }
};

} // namespace detail

template <class T>
inline void SimpleObservableData<T>::read_xml(std::istream& infile, const XMLTag& intag, label_type& l)
{
  detail::input_helper<obs_value_traits<T>::array_valued>::read_xml(*this,infile,intag,l);
}

template <class T> 
std::string SimpleObservableData<T>::evaluation_method(Target t) const
{
  if (t==Variance)
    return "simple";
  else if (eval_method_!="")
    return eval_method_;
  else if (jack_.size())
    return "jackknife";
  else if (has_tau_)
    return "binning";
  else
    return "simple";
}

template <class T> 
SimpleObservableData<T>& SimpleObservableData<T>::operator+=(const SimpleObservableData<T>& x)
{
  using std::sqrt;
  if(count() && x.count()) {
    mean_ += x.mean();
    error_ *= error_;
    error_ += x.error()*x.error();
    error_ = sqrt(error_);
    // error_=sqrt(error_*error_+x.error()*x.error());
  }
  transform(x, _1+_2);
  return (*this);
}

template <class T>
SimpleObservableData<T>& SimpleObservableData<T>::operator-=(const SimpleObservableData<T>& x)
{
  using std::sqrt;
  if(count() && x.count()) {
    mean_ -= x.mean();
    error_ *= error_;
    error_ += x.error()*x.error();
    error_ = sqrt(error_);
    //error_=sqrt(error_*error_+x.error()*x.error());
  }
  transform(x,_1-_2);
  return (*this);
}

template <class T>
template<class X>
SimpleObservableData<T>& SimpleObservableData<T>::operator*=(const SimpleObservableData<X>& x)
{
  using std::sqrt;
  if(count() && x.count()) {
    error_=error()*error();
    error_*=x.mean()*x.mean();
    typename SimpleObservableData<X>::result_type tmp(x.error());
    tmp *=tmp;
    result_type tmp2(mean_);
    tmp2 *= tmp2*tmp;
    error_ += tmp2;
    error_ = sqrt(error_);
    mean_ *= x.mean();
    //error_=sqrt(error()*error()*x.mean()*x.mean()+mean()*mean()*x.error()*x.error());
  }
  transform(x,_1*_2,1./x.bin_size());
  return (*this);
}

template <class T>
template<class X>
SimpleObservableData<T>& SimpleObservableData<T>::operator/=(const SimpleObservableData<X>& x)
{
  using std::sqrt;
  if(count() && x.count()) {
    error_=error()*error();
    typename SimpleObservableData<X>::result_type m(x.mean());
    m *=m;
    typename SimpleObservableData<X>::result_type tmp(x.error());
    tmp *=m;
    tmp *=x.error()*m;
    error_ +=tmp;
    error_ /=m;
    error_ = sqrt(error_);
    mean_ /= x.mean();
    //error_ = sqrt((error()*error()+mean()*mean()*x.error()*x.error()/x.mean()/x.mean())/x.mean()/x.mean());
  }
  transform(x,_1/_2,x.bin_size());
  return (*this);
}

template <class T>
template <class X, class OP>
void SimpleObservableData<T>::transform(const SimpleObservableData<X>& x, OP op, double factor)
{
  if ((count()==0) || (x.count()==0))
    boost::throw_exception(std::runtime_error("both observables need measurements"));
    
  if(!jack_valid_) fill_jack();
  if(!x.jack_valid_) x.fill_jack();

  valid_ = false;
  nonlinear_operations_ = true;
  changed_ = true;
  has_minmax_ = false;
  has_variance_ = false;
  has_tau_=false;
  values2_.clear();
  bool delete_bins = (bin_number() != x.bin_number() ||
                      bin_size() != x.bin_size() ||
                      jack_.size() != x.jack_.size() );
  if (delete_bins) {
    values_.clear();
    jack_.clear();
  } else {
    for (uint64_t i = 0; i < bin_number(); ++i)
      values_[i] = op(values_[i], x.values_[i])*factor;
    for (uint64_t i = 0; i < jack_.size(); ++i)
      jack_[i] = op(jack_[i], x.jack_[i]);
  }
  
  obs_value_traits<convergence_type>::check_for_max(converged_errors_, x.converged_errors());
  obs_value_traits<convergence_type>::check_for_min(any_converged_errors_, x.any_converged_errors());
   
}

template <class T>
ALPS_DUMMY_VOID SimpleObservableData<T>::compact()
{
  analyze();
  count_=count();
  values_.clear();
  values2_.clear();
  jack_.clear();
  ALPS_RETURN_VOID
}

template <class T> 
template <class OP>
void SimpleObservableData<T>::transform_linear(OP op)
{
  mean_ = op(mean_);
  std::transform(values_.begin(), values_.end(), values_.begin(), op);
  fill_jack();
  std::transform(jack_.begin(), jack_.end(), jack_.begin(), op);
}

template <class T> 
template <class OP>
void SimpleObservableData<T>::transform(OP op)
{
  valid_ = false;
  nonlinear_operations_ = true;
  changed_ = true;
  has_variance_ = false;
  has_tau_ = false;
  values2_.clear();
  has_minmax_ = false;
  std::transform(values_.begin(), values_.end(), values_.begin(),_1/double(bin_size()));
  std::transform(values_.begin(), values_.end(), values_.begin(), op);
  std::transform(values_.begin(), values_.end(), values_.begin(),_1*double(bin_size()));
  fill_jack();
  std::transform(jack_.begin(), jack_.end(), jack_.begin(), op);
}

template <class T>
void SimpleObservableData<T>::negate()
{
  if (count()) {
    /*if (has_minmax_) {
      value_type tmp(min_);
      min_ = -max_;
      max_ = -tmp;
    }*/
    transform_linear(-_1);
  }
}

template <class T> template <class X>
SimpleObservableData<T>& SimpleObservableData<T>::operator+=(X x)
{
  if (count()) {
    /*if (has_minmax_) {
      min_ += x;
      max_ += x;
    }*/
    transform_linear(_1 + x);
    for (int i=0;i<values2_.size();++i)
      values2_[i] += 2.*values_[i]*x+x*x;
  }
  return *this;
}

template <class T> template <class X>
SimpleObservableData<T>& SimpleObservableData<T>::operator-=(X x)
{
  if(count()) {
    /*if (has_minmax_) {
      min_ -= x;
      max_ -= x;
    }*/
    transform_linear(_1-x);
    for (int i=0;i<values2_.size();++i)
      values2_[i] += -2.*values_[i]*x+x*x;
  }
  return (*this);
}

template <class T> template <class X>
void SimpleObservableData<T>::subtract_from(const X& x)
{
  if (count()) {
    /*if(has_minmax_) {
      min_ = x-max_;
      max_ = x-min_;
    }*/
    transform_linear(x-_1);
    for (int i=0;i<values2_.size();++i)
      values2_[i] += -2.*values_[i]*x+x*x;
  }
}

template <class T> template <class X>
SimpleObservableData<T>& SimpleObservableData<T>::operator*=(X x)
{
  if (count()) {
    error_ *= x;
    if(has_variance_)
      variance_ *= x*x;
    has_minmax_ = false;
    
    transform_linear(_1*x);
    std::transform(values2_.begin(),values2_.end(),values2_.begin(),_1*(x*x));
  }
  return (*this);
}

template <class T> template <class X>
SimpleObservableData<T>& SimpleObservableData<T>::operator/=(X x)
{
  if (count()) {
    error_ /= x;
    if(has_variance_)
      variance_ /= x*x;
    has_minmax_ = false;
    
    transform_linear(_1/x);
    std::transform(values2_.begin(),values2_.end(),values2_.begin(),_1/(x*x));
  }
  return (*this);
}

template <class T> template <class X>
void SimpleObservableData<T>::divide(const X& x)
{
  if (count()) {
    error_ = x *error_/mean_/mean_;
    has_minmax_ = false;
    has_variance_ = false;
    values2_.clear();
    has_tau_ = false;
    nonlinear_operations_ = true;
    changed_ = true;
	mean_ = x/mean_;
    std::transform(values_.begin(), values_.end(), values_.begin(), (x*bin_size()*bin_size())/_1);
    fill_jack();
    std::transform(jack_.begin(), jack_.end(), jack_.begin(), x/_1);
  }
}

template <class T>
void SimpleObservableData<T>::collect_from(const std::vector<SimpleObservableData<T> >& runs)
{
  bool got_data = false;

  count_ = 0;

  changed_ = false;
  valid_ = false;
  jack_valid_ = false;
  nonlinear_operations_ = false;

  discardedbins_ = 0;
  discardedmeas_ = 0;
  has_variance_ = false;
  has_tau_ = false;
  has_minmax_ = false;

  values_.clear();
  values2_.clear();
  jack_.clear();

  // find smallest and largest bin sizes
  uint64_t minsize = std::numeric_limits<uint64_t>::max();
  uint64_t maxsize = 0;
  for (typename std::vector<SimpleObservableData<T> >::const_iterator
         r = runs.begin(); r != runs.end(); ++r) {
    if (r->count()) {
      if (r->bin_size() < minsize) minsize = r->bin_size();
      if (r->bin_size() > maxsize) maxsize = r->bin_size();
    }
  }

  binsize_ = maxsize;
  
  for (typename std::vector<SimpleObservableData<T> >::const_iterator
         r = runs.begin(); r != runs.end(); ++r) {
    if (r->count()) {
      if (!got_data) {
        // initialize
        jack_valid_ = true;

        has_variance_ = r->has_variance_;
        has_tau_ = r->has_tau_;
        has_minmax_ = false; //r->has_minmax_;
        can_set_thermal_ = r->can_set_thermal_;
        nonlinear_operations_ = r->nonlinear_operations_;
        changed_ = r->changed_;
        obs_value_traits<result_type>::copy(mean_,r->mean_);
        obs_value_traits<result_type>::copy(error_,r->error_);
        obs_value_traits<convergence_type>::copy(converged_errors_,r->converged_errors_);
        obs_value_traits<convergence_type>::copy(any_converged_errors_,r->any_converged_errors_);
        /*if (has_minmax_) {
          obs_value_traits<value_type>::copy(min_, r->min_);
          obs_value_traits<value_type>::copy(max_, r->max_);
        }*/
        if(has_variance_)
          obs_value_traits<result_type>::copy(variance_,r->variance_);
        if(has_tau_)
          obs_value_traits<time_type>::copy(tau_,r->tau_);
        thermalcount_ = r->thermalcount_;
        discardedmeas_ = r->discardedmeas_;
        count_ = r->count();

        if (r->bin_size() == maxsize) {
          r->fill_jack();
          values_ = r->values_;
          values2_ = r->values2_;
          jack_ = r->jack_;
        } else {
          SimpleObservableData<T> tmp(*r);
          tmp.set_bin_size(maxsize);
          tmp.fill_jack();
          values_ = tmp.values_;
          values2_ = tmp.values2_;
          jack_ = tmp.jack_;
        }
        got_data=true;
      } else {
        // add
        jack_valid_ = false;

        has_variance_ = has_variance_ && r->has_variance_;
        has_tau_ = has_tau_ && r->has_tau_;
        has_minmax_ = false; //has_minmax_ && r->has_minmax_;
        can_set_thermal_ = can_set_thermal_ && r->can_set_thermal_;
        nonlinear_operations_ = nonlinear_operations_ || r->nonlinear_operations_;
        changed_ = changed_ && r->changed_;
        /*if(has_minmax_) {
          obs_value_traits<value_type>::check_for_min(min_, r->min_);
          obs_value_traits<value_type>::check_for_max(max_, r->max_);
        }*/
        obs_value_traits<convergence_type>::check_for_max(converged_errors_, r->converged_errors_);
        obs_value_traits<convergence_type>::check_for_min(any_converged_errors_, r->any_converged_errors_);

        mean_ *= double(count_);
        mean_ += double(r->count_)*r->mean_;
        mean_ /= double(count_ + r->count_);
        //mean_ = (double(count_)*mean_+double(r->count_)*r->mean_)
        //        / double(count_ + r->count_);
        using std::sqrt;
        result_type tmp = error_;
        tmp *= error_*(double(count_)*double(count_));
        result_type tmp2 = r->error_;
        tmp2 *= r->error_*(double(r->count_)*double(r->count_));
        error_=tmp+tmp2;
        error_=sqrt(error_);
        error_/=double(count_ + r->count_);
        //error_ = sqrt(double(count_)*double(count_)*error_*error_
         //             +double(r->count_)*double(r->count_)*r->error_*r->error_)
         // / double(count_ + r->count_);
        if(has_variance_) {
          variance_*=double(count_);
          variance_+=double(r->count_)*r->variance_;
          variance_ /= double(count_ + r->count_);
          //variance_ = (double(count_)*variance_+double(r->count_)*r->variance_)
          //  / double(count_ + r->count_);
        }
        if(has_tau_) {
          tau_ *= double(count_);
          tau_ += double(r->count_)*r->tau_;
          tau_ /= double(count_ + r->count_);
          //tau_ = (double(count_)*tau_+double(r->count_)*r->tau_)
          //  / double(count_ + r->count_);
        }

        thermalcount_ = std::min(thermalcount_, r->thermalcount_);
        discardedmeas_ = std::min(discardedmeas_, r->discardedmeas_);
        count_ += r->count();

        if (r->bin_size() == maxsize) {
          std::copy(r->values_.begin(), r->values_.end(),
                    std::back_inserter(values_));
          std::copy(r->values2_.begin(), r->values2_.end(),
                    std::back_inserter(values2_));
        } else {
          SimpleObservableData<T> tmp(*r);
          tmp.set_bin_size(maxsize);
          std::copy(tmp.values_.begin(), tmp.values_.end(),
                    std::back_inserter(values_));
          std::copy(tmp.values2_.begin(), tmp.values2_.end(),
                    std::back_inserter(values2_));
        }
      }
    }
  }

  analyze();
}

#ifndef ALPS_WITHOUT_OSIRIS

template <class T>
void SimpleObservableData<T>::extract_timeseries(ODump& dump) const
{
  dump << binsize_ << uint64_t(values_.size()) << binsize_ << values_;
}

template <class T>
void SimpleObservableData<T>::save(ODump& dump) const
{
  dump << count_ << mean_ << error_ << variance_ << tau_ << has_variance_
       << has_tau_ << has_minmax_ << thermalcount_ << can_set_thermal_ /*<< min_ << max_*/
       << binsize_ << discardedmeas_ << discardedbins_ << valid_ << jack_valid_ << changed_
       << nonlinear_operations_ << values_ << values2_ << jack_ << converged_errors_ << any_converged_errors_;
}

template <class T>
void SimpleObservableData<T>::load(IDump& dump)
{
  if(dump.version() >= 305 || dump.version() == 0 /* version is not set */){
    dump >> count_ >> mean_ >> error_ >> variance_ >> tau_ >> has_variance_
         >> has_tau_ >> has_minmax_ >> thermalcount_ >> can_set_thermal_ 
         >> binsize_ >> discardedmeas_ >> discardedbins_ >> valid_ >> jack_valid_ >> changed_
         >> nonlinear_operations_ >> values_ >> values2_ >> jack_;
  }
  else if(dump.version() >= 302 ){
    value_type min_ignored, max_ignored;
    dump >> count_ >> mean_ >> error_ >> variance_ >> tau_ >> has_variance_
         >> has_tau_ >> has_minmax_ >> thermalcount_ >> can_set_thermal_ 
         >> min_ignored >> max_ignored
         >> binsize_ >> discardedmeas_ >> discardedbins_ >> valid_ >> jack_valid_ >> changed_
         >> nonlinear_operations_ >> values_ >> values2_ >> jack_;
  }
  else {
    value_type min_ignored, max_ignored;
    // some data types have changed from 32 to 64 Bit between version 301 and 302
    uint32_t count_tmp, binsize_tmp;
    dump >> count_tmp >> mean_ >> error_ >> variance_ >> tau_ >> has_variance_
         >> has_tau_ >> has_minmax_ >> thermalcount_ >> can_set_thermal_
         >> min_ignored >> max_ignored
         >> binsize_tmp >> discardedmeas_ >> discardedbins_ >> valid_ >> jack_valid_ >> changed_
         >> nonlinear_operations_ >> values_ >> values2_ >> jack_;
    // perform the conversions which may be necessary
    count_ = count_tmp;
    binsize_ = binsize_tmp;
   }
  if (dump.version() > 300 || dump.version() == 0 /* version is not set */)
    dump >> converged_errors_ >> any_converged_errors_;
}

#endif

template <class T>
void SimpleObservableData<T>::fill_jack() const
{
  // build jackknife data structure
  if (bin_number() && !jack_valid_) {
    if (nonlinear_operations_)
      boost::throw_exception(std::runtime_error("Cannot rebuild jackknife data structure after nonlinear operations"));
    jack_.clear();
    jack_.resize(bin_number() + 1);

    // Order-N initialization of jackknife data structure
    obs_value_traits<result_type>::resize_same_as(jack_[0], bin_value(0));
    for(uint64_t i = 0; i < bin_number(); ++i) 
      jack_[0] += obs_value_traits<result_type>::convert(bin_value(i)) / count_type(bin_size());
    for(uint64_t i = 0; i < bin_number(); ++i) {
      obs_value_traits<result_type>::resize_same_as(jack_[i+1], jack_[0]);
      result_type tmp(obs_value_traits<result_type>::convert(bin_value(i)));
      tmp /= count_type(bin_size());
      jack_[i+1] = jack_[0]
          - tmp;
//        - (obs_value_traits<result_type>::convert(bin_value(i)) / count_type(bin_size()));
      jack_[i+1] /= count_type(bin_number() - 1);
    }
    jack_[0] /= count_type(bin_number());
  }
  jack_valid_ = true;
}

template <class T>
void SimpleObservableData<T>::analyze() const
{
  if (valid_) return;

  if (bin_number())
  {
    count_ = bin_size()*bin_number();

//    std::cerr<<"running jackknife. "<<std::endl;
    // calculate mean and error
    jackknife();

    // calculate variance and tau
    if (!values2_.empty()) {
      has_variance_ = true;
      has_tau_ = true;
      obs_value_traits<result_type>::resize_same_as(variance_, bin_value2(0));
      variance_ = 0.;
      for (uint64_t i=0;i<values2_.size();++i)
        variance_+=obs_value_traits<result_type>::convert(values2_[i]);
      // was: variance_ = std::accumulate(values2_.begin(), values2_.end(), variance_);
      result_type mean2(mean_);
      mean2*=mean_*count_type(count());
      variance_ -= mean2;
      variance_ /= count_type(count()-1);
      obs_value_traits<result_type>::resize_same_as(tau_, error_);
      tau_=error_;
      tau_*=error_*count_type(count());
      tau_/=variance_;
      tau_-=1.;
      tau_*=0.5;
    } else {
      has_variance_ = false;
      has_tau_ = false;
    }
  }
  valid_ = true;
}

template <class T>
void SimpleObservableData<T>::jackknife() const
{
  fill_jack();

  if (jack_.size()) {
    // if any run is converged the errors will be OK
    converged_errors_=any_converged_errors_;
    
    result_type rav;
    obs_value_traits<result_type>::resize_same_as(mean_, jack_[0]);  
    obs_value_traits<result_type>::resize_same_as(error_, jack_[0]);  
    obs_value_traits<result_type>::resize_same_as(rav, jack_[0]);  
    unsigned int k = jack_.size()-1;

    rav = 0;
    rav = std::accumulate(jack_.begin()+1, jack_.end(), rav);
    rav /= count_type(k);
    
    result_type tmp(rav);
    tmp -= jack_[0];
    tmp *= count_type(k - 1);
    mean_  = jack_[0]-tmp;
    //mean_ = jack_[0] - (rav - jack_[0]) * count_type(k - 1);

    error_ = 0.0;
    for (unsigned int i = 1; i < jack_.size(); ++i)
      error_ += (jack_[i] - rav) * (jack_[i] - rav);
      //error_ += jack_[i] * jack_[i];
    
    error_/=count_type(k);
    //error_-= rav * rav;
    //error_ = (error_ / count_type(k) - rav * rav);
    error_ *= count_type(k - 1);
    error_ = std::sqrt(error_);
  }
}


template<class T>
typename SimpleObservableData<T>::covariance_type 
SimpleObservableData<T>::covariance(const SimpleObservableData<T> obs2) const
{
  fill_jack();
  obs2.fill_jack();
  if (jack_.size() && obs2.jack_.size()) {
    result_type rav1;
    result_type rav2;
    obs_value_traits<result_type>::resize_same_as(rav1, jack_[0]);  
    obs_value_traits<result_type>::resize_same_as(rav2, obs2.jack_[0]);  
    if (jack_.size() != obs2.jack_.size()) 
      boost::throw_exception(std::runtime_error("unequal number of bins in calculation of covariance matrix"));
    uint32_t k = jack_.size()-1;

    rav1 = 0;
    rav2 = 0;
    rav1 = std::accumulate(jack_.begin()+1, jack_.end(), rav1);
    rav2 = std::accumulate(obs2.jack_.begin()+1, obs2.jack_.end(), rav2);
    rav1 /= count_type(k);
    rav2 /= count_type(k);
    
    covariance_type cov = obs_value_traits<T>::outer_product(jack_[1],obs2.jack_[1]);
    for (uint32_t i = 2; i < jack_.size(); ++i)
      cov += obs_value_traits<T>::outer_product(jack_[i],obs2.jack_[i]);
    
    cov/=count_type(k);
    cov-= obs_value_traits<T>::outer_product(rav1, rav2);
    cov *= count_type(k - 1);
    return cov;
  } else {
    boost::throw_exception(std::runtime_error ("no binning information available for calculation of covariances"));
    covariance_type dummy;
    return dummy;
  }
}


template <class T>
uint32_t SimpleObservableData<T>::get_thermalization() const
{
  return thermalcount_ + discardedmeas_;
}

template <class T>
ALPS_DUMMY_VOID SimpleObservableData<T>::set_thermalization(uint32_t thermal)
{
  if (nonlinear_operations_)
    boost::throw_exception(std::runtime_error("cannot set thermalization after nonlinear operations"));
  if (!can_set_thermalization())
    boost::throw_exception(std::runtime_error("cannot set thermalization"));
  if (binsize_) {
    discardedmeas_ = thermal - thermalcount_;
    discardedbins_ = (discardedmeas_ + binsize_ - 1) / binsize_;  
    changed_ = true;
    valid_ = false;
    jack_valid_ = false;
  }
  ALPS_RETURN_VOID
}

template <class T>
void SimpleObservableData<T>::collect_bins(uint64_t howmany)
{
  if (nonlinear_operations_)
    boost::throw_exception(std::runtime_error("cannot change bins after nonlinear operations"));
  if (values_.empty() || howmany <= 1) return;
    
  uint64_t newbins = values_.size() / howmany;
  
  // fill bins
  for (uint64_t i = 0; i < newbins; ++i) {
    values_[i] = values_[howmany * i];
    if (!values2_.empty()) values2_[i] = values2_[howmany * i];
    for (uint64_t j = 1; j < howmany; ++j) {
      values_[i] += values_[howmany * i + j];
      if (!values2_.empty()) values2_[i] += values2_[howmany * i + j];
    }
  }
  
  binsize_ *= howmany;
  discardedbins_ = (discardedmeas_ + binsize_ - 1) / binsize_;

  values_.resize(newbins);
  if (!values2_.empty()) values2_.resize(newbins);
  
  changed_ = true;
  jack_valid_ = false;
  valid_ = false;
}

template <class T>
void SimpleObservableData<T>::set_bin_size(uint64_t s)
{
  collect_bins((s-1)/binsize_+1);
  binsize_=s;
}

template <class T>
void SimpleObservableData<T>::set_bin_number(uint64_t binnum)
{
  collect_bins((values_.size()-1)/binnum+1);
}

template <class T>
const typename SimpleObservableData<T>::result_type& SimpleObservableData<T>::mean() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  analyze();
  return mean_;
}

template <class T>
const typename SimpleObservableData<T>::result_type& SimpleObservableData<T>::error() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
//  std::cerr<<"analyzing."<<std::endl;
  analyze();
  return error_;
}

template <class T>
const typename SimpleObservableData<T>::convergence_type& SimpleObservableData<T>::converged_errors() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  analyze();
  return converged_errors_;
}

template <class T>
const typename SimpleObservableData<T>::convergence_type& SimpleObservableData<T>::any_converged_errors() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  analyze();
  return any_converged_errors_;
}

template <class T>
const typename SimpleObservableData<T>::result_type& SimpleObservableData<T>::variance() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  if (!has_variance_)
    boost::throw_exception(std::logic_error("observable does not have variance"));
  analyze();
  return variance_;
}

template <class T>
inline
const typename SimpleObservableData<T>::time_type& SimpleObservableData<T>::tau() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  if (!has_tau_)
    boost::throw_exception(std::logic_error("observable does not have autocorrelation information"));
  analyze();
  return tau_;
}

/*template <class T>
inline
const typename SimpleObservableData<T>::value_type& SimpleObservableData<T>::min() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  //if (!has_minmax_)
    boost::throw_exception(std::logic_error("observable does not have minimum"));
  //return min_;
}

template <class T>
inline
const typename SimpleObservableData<T>::value_type& SimpleObservableData<T>::max() const
{
  if (count() == 0) boost::throw_exception(NoMeasurementsError());
  //if(!has_minmax_)
    boost::throw_exception(std::logic_error("observable does not have maximum"));
  //return max_;
}*/

} // end namespace alps


//
// OSIRIS support
//

#ifndef ALPS_WITHOUT_OSIRIS

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template<class T>
inline alps::ODump& operator<<(alps::ODump& od, const alps::SimpleObservableData<T>& m)
{ m.save(od); return od; }

template<class T>
inline alps::IDump& operator>>(alps::IDump& id, alps::SimpleObservableData<T>& m)
{ m.load(id); return id; }

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif // !ALPS_WITHOUT_OSIRIS

#endif // ALPS_ALEA_SIMPLEOBSDATA_H
