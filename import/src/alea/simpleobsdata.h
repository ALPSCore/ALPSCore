/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_ALEA_SIMPLEOBSDATA_H
#define ALPS_ALEA_SIMPLEOBSDATA_H

#include <alps/config.h>
#include <alps/alea/nan.h>
#include <alps/alea/convergence.hpp>
#include <alps/alea/simpleobservable.h>
#include <alps/parser/parser.h>
#include <alps/osiris/std/valarray.h>
#include <alps/type_traits/is_scalar.hpp>
#include <alps/type_traits/change_value_type.hpp>
#include <alps/type_traits/average_type.hpp>
#include <alps/type_traits/covariance_type.hpp>
#include <alps/numeric/update_minmax.hpp>
#include <alps/numeric/outer_product.hpp>
#include <alps/utilities/numeric_cast.hpp>
#include <alps/utilities/resize.hpp>
#include <alps/utilities/assign.hpp>
#include <alps/lambda.hpp>
#include <alps/hdf5.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/functional.hpp>

#include <iostream>
#include <numeric>
#include <vector>
#include <boost/config.hpp>
#include <valarray>

template <class T> std::ostream& operator<<(std::ostream& o, const std::valarray<T>&) { return o;}

namespace alps {

class RealObsevaluatorXMLHandler;
class RealVectorObsevaluatorXMLHandler;


//=======================================================================
// SimpleObservableData
//
// Observable class with variable autocorrelation analysis and binning
//-----------------------------------------------------------------------


template <class T>
class SimpleObservableData {
public:
  template <class X>
  friend class SimpleObservableData;

  friend class RealObsevaluatorXMLHandler;
  friend class RealVectorObsevaluatorXMLHandler;

  typedef T value_type;
  typedef typename change_value_type<T,double>::type time_type;
  typedef std::size_t size_type;
  typedef double count_type;
  typedef typename average_type<T>::type result_type;
  typedef typename change_value_type<T,int>::type convergence_type;
  typedef typename change_value_type_replace_valarray<value_type,std::string>::type label_type;
  typedef typename covariance_type<T>::type covariance_type;

  // constructors
  SimpleObservableData();
  template <class U, class S>
  SimpleObservableData(const SimpleObservableData<U>& x, S s);
  SimpleObservableData(const AbstractSimpleObservable<value_type>& obs);
  SimpleObservableData(std::istream&, const XMLTag&, label_type& );

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

  covariance_type covariance(const SimpleObservableData<T>) const;

  bool has_variance() const { return has_variance_;}
  bool has_tau() const { return has_tau_;}

  uint64_t bin_size() const { return binsize_;}
  std::size_t bin_number() const { return values_.size()-discardedbins_;}
  std::size_t bin_number2() const { return discardedbins_ ? 0 : values2_.size();}
  const value_type& bin_value(std::size_t i) const {
    return values_[i+discardedbins_];
  }
  const value_type& bin_value2(std::size_t i) const {
    return values2_[i+discardedbins_];
  }

  const std::vector<value_type>& bins() const {
    return values_;
  }

  template <class S>
  SimpleObservableData<typename element_type<T>::type> slice(S s) const
  {
    return SimpleObservableData<typename element_type<T>::type>(*this,s);
  }

  ALPS_DUMMY_VOID compact();

  void extract_timeseries(ODump& dump) const;
  void save(ODump& dump) const;
  void load(IDump& dump);

  void save(hdf5::archive &) const;
  void load(hdf5::archive &);

  inline void set_bin_size(uint64_t);
  inline void set_bin_number(std::size_t);

  SimpleObservableData<T> &  operator<<(const SimpleObservableData<T>& b);

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
  void collect_bins(std::size_t howmany);
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
  mutable bool can_set_thermal_;

  mutable uint64_t binsize_;
  mutable uint64_t max_bin_number_;
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
   binsize_(0),
   max_bin_number_(0),
   discardedmeas_(0),
   discardedbins_(0),
   changed_(false),
   valid_(true),
   jack_valid_(true),
   nonlinear_operations_(false),
   mean_(),
   error_(),
   variance_(),
   tau_(),
   values_(),
   values2_(),
   jack_(),
   converged_errors_(),
   any_converged_errors_()
{}

template <class T>
template <class U, class S>
inline
SimpleObservableData<T>::SimpleObservableData(const SimpleObservableData<U>& x, S s)
 : count_(x.count_),
   has_variance_(x.has_variance_),
   has_tau_(x.has_tau_),
   can_set_thermal_(x.can_set_thermal_),
   binsize_(x.binsize_),
   max_bin_number_(x.max_bin_number_),
   discardedmeas_(x.discardedmeas_),
   discardedbins_(x.discardedbins_),
   changed_(x.changed_),
   valid_(x.valid_),
   jack_valid_(x.jack_valid_),
   nonlinear_operations_(x.nonlinear_operations_),
   mean_(slice_value(x.mean_, s)),
   error_(slice_value(x.error_, s)),
   variance_(has_variance_ ? slice_value(x.variance_, s) : result_type()),
   tau_(has_tau_ ? slice_value(x.tau_, s) : time_type()),
   values_(x.values_.size()),
   values2_(x.values2_.size()),
   jack_(x.jack_.size()),
   converged_errors_(slice_value(x.converged_errors_,s)),
   any_converged_errors_(slice_value(x.any_converged_errors_,s))
{
  std::transform(x.values_.begin(), x.values_.end(), values_.begin(),
                 boost::bind2nd(slice_it<U>(),s));
  std::transform(x.values2_.begin(), x.values2_.end(), values2_.begin(),
                 boost::bind2nd(slice_it<U>(),s));
  if (jack_valid_) {
    std::transform(x.jack_.begin(), x.jack_.end(), jack_.begin(),
                   boost::bind2nd(slice_it<U>(),s));
  }
}


template <class T>
inline
SimpleObservableData<T> const& SimpleObservableData<T>::operator=(const SimpleObservableData<T>& x)
 {
   count_=x.count_;
   has_variance_=x.has_variance_;
   has_tau_=x.has_tau_;
   can_set_thermal_=x.can_set_thermal_;
   binsize_=x.binsize_;
   max_bin_number_=x.max_bin_number_;
   discardedmeas_=x.discardedmeas_;
   discardedbins_=x.discardedbins_;
   changed_=x.changed_;
   valid_=x.valid_;
   jack_valid_=x.jack_valid_;
   nonlinear_operations_=x.nonlinear_operations_;
   assign(mean_,x.mean_);
   assign(error_,x.error_);
   assign(variance_,x.variance_);
   assign(tau_,x.tau_);
   values_=x.values_;
   values2_=x.values2_;
   jack_=x.jack_;

   assign(converged_errors_,x.converged_errors_);
   assign(any_converged_errors_,x.any_converged_errors_);

  return *this;
}


template <class T>
SimpleObservableData<T>::SimpleObservableData(const AbstractSimpleObservable<T>& obs)
 : count_(obs.count()),
   has_variance_(obs.has_variance()),
   has_tau_(obs.has_tau()),
   can_set_thermal_(true),
   binsize_(obs.bin_size()),
   max_bin_number_(obs.max_bin_number()),
   discardedmeas_(0),
   discardedbins_(0),
   changed_(false),
   valid_(false),
   jack_valid_(false),
   nonlinear_operations_(false),
   mean_(),
   error_(),
   variance_(),
   tau_(),
   values_(),
   values2_(),
   jack_()
{
  if (count()) {
    assign(mean_, obs.mean());
    assign(error_, obs.error());
    if (has_variance())
      assign(variance_, obs.variance());
    if (has_tau())
      assign(tau_, obs.tau());

    for (std::size_t i = 0; i < obs.bin_number(); ++i)
      values_.push_back(obs.bin_value(i));
    for (std::size_t i = 0; i < obs.bin_number2(); ++i)
      values2_.push_back(obs.bin_value2(i));
    assign(converged_errors_, obs.converged_errors());
    assign(any_converged_errors_, obs.converged_errors());

    if (bin_size() != 1 && bin_number() > max_bin_number_) set_bin_number(max_bin_number_);
  }
}

template <class T>
SimpleObservableData<T>::SimpleObservableData(std::istream& infile, const XMLTag& intag, label_type& l)
  : count_(0),
    has_variance_(false),
    has_tau_(false),
    can_set_thermal_(false),
    binsize_(0),
    max_bin_number_(0),
    discardedmeas_(0),
    discardedbins_(0),
    changed_(false),
    valid_(true),
    jack_valid_(false),
    nonlinear_operations_(false),
    mean_(),
    error_(),
    variance_(),
    tau_(),
    values_(),
    values2_(),
    jack_()
{
  read_xml(infile,intag,l);
}

inline double text_to_double(const std::string& val)
{
  return ((val=="NaN" || val=="nan" || val=="NaNQ" || val == "-nan") ? alps::nan() :
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
  mean_.resize(s);
  error_.resize(s);
  variance_.resize(s);
  tau_.resize(s);
  converged_errors_.resize(s);
  any_converged_errors_.resize(s);
  label.resize(s);

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

template <class F> struct input_helper {};

template <> struct input_helper<boost::mpl::false_>
{
  template <class T, class L>
  static void read_xml(SimpleObservableData<T>& obs, std::istream& infile, const XMLTag& tag, L& l) {
    obs.read_xml_vector(infile,tag,l);
  }
};

template <> struct input_helper<boost::mpl::true_>
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
  detail::input_helper<typename is_scalar<T>::type>::read_xml(*this,infile,intag,l);
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
  transform(x, boost::lambda::_1+boost::lambda::_2);
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
  transform(x,boost::lambda::_1-boost::lambda::_2);
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
  transform(x,boost::lambda::_1*boost::lambda::_2,1./x.bin_size());
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
  transform(x,boost::lambda::_1/boost::lambda::_2,x.bin_size());
  return (*this);
}

template <class T>
template <class X, class OP>
void SimpleObservableData<T>::transform(const SimpleObservableData<X>& x, OP op, double factor)
{
  if ((count()==0) || (x.count()==0))
    boost::throw_exception(std::runtime_error("both observables need measurements"));

  bool delete_bins = (bin_number() != x.bin_number() ||
                      bin_size() != x.bin_size());
  
  if (delete_bins) {
    std::cerr << "Bin number: " << bin_number() << " " << x.bin_number() << "\n";
    std::cerr << "Bin size:   " << bin_size() << " " << x.bin_size() << "\n";
    boost::throw_exception(std::runtime_error("both observables need same number of measurements and bins"));
  }

  if(!jack_valid_) fill_jack();
  if(!x.jack_valid_) x.fill_jack();
  

  valid_ = false;
  nonlinear_operations_ = true;
  changed_ = true;
  has_variance_ = false;
  has_tau_=false;
  values2_.clear();
  if (delete_bins) {
    values_.clear();
    jack_.clear();
  } else {
    for (std::size_t i = 0; i < bin_number(); ++i)
      values_[i] = op(values_[i], x.values_[i])*factor;
    for (std::size_t i = 0; i < jack_.size(); ++i)
      jack_[i] = op(jack_[i], x.jack_[i]);
  }
}

template <class T>
void SimpleObservableData<T>::compact()
{
  analyze();
  count_=count();
  values_.clear();
  values2_.clear();
  jack_.clear();
}

template <class T>
template <class OP>
void SimpleObservableData<T>::transform_linear(OP op)
{
  fill_jack();
  mean_ = op(mean_);
  std::transform(values_.begin(), values_.end(), values_.begin(), op);
  std::transform(jack_.begin(), jack_.end(), jack_.begin(), op);
}

template <class T>
template <class OP>
void SimpleObservableData<T>::transform(OP op)
{
  fill_jack();
  valid_ = false;
  nonlinear_operations_ = true;
  changed_ = true;
  has_variance_ = false;
  has_tau_ = false;
  values2_.clear();
  std::transform(values_.begin(), values_.end(), values_.begin(),boost::lambda::_1/double(bin_size()));
  std::transform(values_.begin(), values_.end(), values_.begin(), op);
  std::transform(values_.begin(), values_.end(), values_.begin(),boost::lambda::_1*double(bin_size()));
  std::transform(jack_.begin(), jack_.end(), jack_.begin(), op);
}

template <class T>
void SimpleObservableData<T>::negate()
{
  if (count())
    transform_linear(-boost::lambda::_1);
}

template <class T> template <class X>
SimpleObservableData<T>& SimpleObservableData<T>::operator+=(X x)
{
  if (count()) {
    transform_linear(boost::lambda::_1 + x);
    for (std::size_t i=0;i<values2_.size();++i)
      values2_[i] += 2.*values_[i]*x+x*x;
  }
  return *this;
}

template <class T> template <class X>
SimpleObservableData<T>& SimpleObservableData<T>::operator-=(X x)
{
  if(count()) {
    transform_linear(boost::lambda::_1-x);
    for (std::size_t i=0;i<values2_.size();++i)
      values2_[i] += -2.*values_[i]*x+x*x;
  }
  return (*this);
}

template <class T> template <class X>
void SimpleObservableData<T>::subtract_from(const X& x)
{
  if (count()) {
    transform_linear(x-boost::lambda::_1);
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

    transform_linear(boost::lambda::_1*x);
    std::transform(values2_.begin(),values2_.end(),values2_.begin(),boost::lambda::_1*(x*x));
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

    transform_linear(boost::lambda::_1/x);
    std::transform(values2_.begin(),values2_.end(),values2_.begin(),boost::lambda::_1/(x*x));
  }
  return (*this);
}

template <class T> template <class X>
void SimpleObservableData<T>::divide(const X& x)
{
  if (count()) {
    fill_jack();
    error_ = x *error_/mean_/mean_;
    has_variance_ = false;
    values2_.clear();
    has_tau_ = false;
    changed_ = true;
    mean_ = x/mean_;
    double f = bin_size() * bin_size();
    std::transform(values_.begin(), values_.end(), values_.begin(), (x*f)/boost::lambda::_1);
    std::transform(jack_.begin(), jack_.end(), jack_.begin(), x/boost::lambda::_1);
    nonlinear_operations_ = true;
  }
}

template <class T> SimpleObservableData<T> & SimpleObservableData<T>::operator<<(const SimpleObservableData<T>& run) {
  if (run.count()) {
    if (!count()) {
      // initialize
      valid_ = false;
      jack_valid_ = true;
      nonlinear_operations_ = false;
      discardedbins_ = 0;
      binsize_ = run.bin_size();
      max_bin_number_ = run.max_bin_number_;
      has_variance_ = run.has_variance_;
      has_tau_ = run.has_tau_;
      can_set_thermal_ = run.can_set_thermal_;
      nonlinear_operations_ = run.nonlinear_operations_;
      changed_ = run.changed_;
      assign(mean_,run.mean_);
      assign(error_,run.error_);
      assign(converged_errors_,run.converged_errors_);
      assign(any_converged_errors_,run.any_converged_errors_);
      if(has_variance_)
        assign(variance_,run.variance_);
      if(has_tau_)
        assign(tau_,run.tau_);
      discardedmeas_ = run.discardedmeas_;
      count_ = run.count();

      run.fill_jack();
      values_ = run.values_;
      values2_ = run.values2_;
      jack_ = run.jack_;
    } else {
      // add
      jack_valid_ = false;
      
      has_variance_ = has_variance_ && run.has_variance_;
      has_tau_ = has_tau_ && run.has_tau_;
      can_set_thermal_ = can_set_thermal_ && run.can_set_thermal_;
      nonlinear_operations_ = nonlinear_operations_ || run.nonlinear_operations_;
      changed_ = changed_ || run.changed_;
      numeric::update_max(converged_errors_, run.converged_errors_);
      numeric::update_min(any_converged_errors_, run.any_converged_errors_);

      mean_ *= double(count_);
      mean_ += double(run.count_)*run.mean_;
      mean_ /= double(count_ + run.count_);
      //mean_ = (double(count_)*mean_+double(run.count_)*run.mean_)
      //        / double(count_ + run.count_);
      using std::sqrt;
      result_type tmp = error_;
      tmp *= error_*(double(count_)*double(count_));
      result_type tmp2 = run.error_;
      tmp2 *= run.error_*(double(run.count_)*double(run.count_));
      error_=tmp+tmp2;
      error_=sqrt(error_);
      error_/=double(count_ + run.count_);
      //error_ = sqrt(double(count_)*double(count_)*error_*error_
      //             +double(run.count_)*double(run.count_)*run.error_*run.error_)
      // / double(count_ + run.count_);
      if(has_variance_) {
        variance_*=double(count_);
        variance_+=double(run.count_)*run.variance_;
        variance_ /= double(count_ + run.count_);
        //variance_ = (double(count_)*variance_+double(run.count_)*run.variance_)
        //  / double(count_ + run.count_);
      }
      if(has_tau_) {
        tau_ *= double(count_);
        tau_ += double(run.count_)*run.tau_;
        tau_ /= double(count_ + run.count_);
        //tau_ = (double(count_)*tau_+double(run.count_)*run.tau_)
        //  / double(count_ + run.count_);
      }

      discardedmeas_ = std::min BOOST_PREVENT_MACRO_SUBSTITUTION (discardedmeas_, run.discardedmeas_);
      max_bin_number_ = std::max BOOST_PREVENT_MACRO_SUBSTITUTION (max_bin_number_, run.max_bin_number_);
      count_ += run.count();

      if (binsize_ <= run.bin_size()) {
        if (binsize_ < run.bin_size())
          set_bin_size(run.bin_size());
        std::copy(run.values_.begin(), run.values_.end(),
                  std::back_inserter(values_));
        std::copy(run.values2_.begin(), run.values2_.end(),
                  std::back_inserter(values2_));
      } else {
        SimpleObservableData<T> tmp(run);
        tmp.set_bin_size(binsize_);
        std::copy(tmp.values_.begin(), tmp.values_.end(),
                  std::back_inserter(values_));
        std::copy(tmp.values2_.begin(), tmp.values2_.end(),
                  std::back_inserter(values2_));
      }
      if (max_bin_number_ && max_bin_number_ < bin_number())
        set_bin_number(max_bin_number_);
    }
  }
  return *this;
}


template <class T>
void SimpleObservableData<T>::extract_timeseries(ODump& dump) const
{
  dump << binsize_ << uint64_t(values_.size()) << binsize_ << values_;
}

template <class T>
void SimpleObservableData<T>::save(ODump& dump) const
{
  dump << count_ << mean_ << error_ << variance_ << tau_ << has_variance_
       << has_tau_ << can_set_thermal_
       << binsize_ << discardedmeas_ << discardedbins_ << valid_ << jack_valid_ << changed_
       << nonlinear_operations_ << values_ << values2_ << jack_ << converged_errors_ 
       << any_converged_errors_ << max_bin_number_;
}
template <class T>
void SimpleObservableData<T>::load(IDump& dump)
{
  // local variables for depreacted members
  bool has_minmax_;
  value_type min_, max_;
  uint32_t thermalcount_;

  if(dump.version() >= 306 || dump.version() == 0 /* version is not set */){
    dump >> count_ >> mean_ >> error_ >> variance_ >> tau_ >> has_variance_
         >> has_tau_ >> can_set_thermal_
         >> binsize_ >> discardedmeas_ >> discardedbins_ >> valid_ >> jack_valid_ >> changed_
         >> nonlinear_operations_ >> values_ >> values2_ >> jack_;
  } else if(dump.version() >= 302 || dump.version() == 0 /* version is not set */){
    dump >> count_ >> mean_ >> error_ >> variance_ >> tau_ >> has_variance_
         >> has_tau_ >> has_minmax_ >> thermalcount_ >> can_set_thermal_
         >> min_ >> max_
         >> binsize_ >> discardedmeas_ >> discardedbins_ >> valid_ >> jack_valid_ >> changed_
         >> nonlinear_operations_ >> values_ >> values2_ >> jack_;
  } else {
    // some data types have changed from 32 to 64 Bit between version 301 and 302
    uint32_t count_tmp, binsize_tmp;
    dump >> count_tmp >> mean_ >> error_ >> variance_ >> tau_ >> has_variance_
         >> has_tau_ >> has_minmax_ >> thermalcount_ >> can_set_thermal_
         >> min_>> max_
         >> binsize_tmp >> discardedmeas_ >> discardedbins_ >> valid_ >> jack_valid_ >> changed_
         >> nonlinear_operations_ >> values_ >> values2_ >> jack_;
    // perform the conversions which may be necessary
    count_ = count_tmp;
    binsize_ = binsize_tmp;
   }
  if (dump.version() > 300 || dump.version() == 0 /* version is not set */)
    dump >> converged_errors_ >> any_converged_errors_;
  if (dump.version() >= 400 || dump.version() == 0)
    dump >> max_bin_number_;
}

template <typename T> void SimpleObservableData<T>::save(hdf5::archive & ar) const {
    analyze();
    ar
        << make_pvp("count", count_)
        << make_pvp("@changed", changed_)
        << make_pvp("@nonlinearoperations", nonlinear_operations_)
    ;
    if (valid_) {
        ar
            << make_pvp("mean/value", mean_)
            << make_pvp("mean/error", error_)
            << make_pvp("mean/error_convergence", converged_errors_)
        ;
        if (has_variance_)
            ar
                << make_pvp("variance/value", variance_)
            ;
        if (has_tau_)
            ar
                << make_pvp("tau/value", tau_)
            ;
        ar
            << make_pvp("timeseries/data", values_)
            << make_pvp("timeseries/data/@discard", discardedbins_)
            << make_pvp("timeseries/data/@maxbinnum", max_bin_number_)
            << make_pvp("timeseries/data/@binningtype", "linear")
            
            << make_pvp("timeseries/data2", values2_)
            << make_pvp("timeseries/data2/@discard", discardedbins_)
            << make_pvp("timeseries/data/@maxbinnum", max_bin_number_)
            << make_pvp("timeseries/data2/@binningtype", "linear")
        ;
        if (jack_valid_)
            ar
                << make_pvp("jacknife/data", jack_)
                << make_pvp("jacknife/data/@binningtype", "linear")
            ;
    }
}
template <typename T> void SimpleObservableData<T>::load(hdf5::archive & ar) {
    can_set_thermal_ = false;
    discardedmeas_ = 0;
    ar
        >> make_pvp("count", count_)
        >> make_pvp("@changed", changed_)
        >> make_pvp("@nonlinearoperations", nonlinear_operations_)
    ;
    if ((valid_ = ar.is_data("mean/value"))) {
        ar
            >> make_pvp("mean/value", mean_)
            >> make_pvp("mean/error", error_)
            >> make_pvp("mean/error_convergence", converged_errors_)
        ;
        if ((has_variance_ = ar.is_data("variance/value")))
            ar
                >> make_pvp("variance/value", variance_)
            ;
        if ((has_tau_ = ar.is_data("tau/value")))
            ar
                >> make_pvp("tau/value", tau_)
            ;
        ar
            >> make_pvp("timeseries/data", values_)
            >> make_pvp("timeseries/data/@discard", discardedbins_)
            >> make_pvp("timeseries/data/@maxbinnum", max_bin_number_)
            >> make_pvp("timeseries/data2", values2_)
        ;
        if ((jack_valid_ = ar.is_data("jacknife/data")))
            ar
                >> make_pvp("jacknife/data", jack_)
            ;
    }
}

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
    resize_same_as(jack_[0], bin_value(0));
    for(std::size_t i = 0; i < bin_number(); ++i)
      jack_[0] += alps::numeric_cast<result_type>(bin_value(i)) / count_type(bin_size());
    for(std::size_t i = 0; i < bin_number(); ++i) {
      resize_same_as(jack_[i+1], jack_[0]);
      result_type tmp(alps::numeric_cast<result_type>(bin_value(i)));
      tmp /= count_type(bin_size());
      jack_[i+1] = jack_[0]
          - tmp;
//        - (alps::numeric_cast<result_type>(bin_value(i)) / count_type(bin_size()));
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

    // calculate mean and error
    jackknife();

    // calculate variance and tau
    if (!values2_.empty()) {
      has_variance_ = true;
      has_tau_ = true;
      resize_same_as(variance_, bin_value2(0));
      variance_ = 0.;
      for (std::size_t i=0;i<values2_.size();++i)
        variance_+=alps::numeric_cast<result_type>(values2_[i]);
      // was: variance_ = std::accumulate(values2_.begin(), values2_.end(), variance_);
      result_type mean2(mean_);
      mean2*=mean_*count_type(count());
      variance_ -= mean2;
      variance_ /= count_type(count()-1);
      resize_same_as(tau_, error_);
      tau_=std::abs(error_);
      tau_*=std::abs(error_)*count_type(count());
      tau_/=std::abs(variance_);
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
    resize_same_as(mean_, jack_[0]);
    resize_same_as(error_, jack_[0]);
    resize_same_as(rav, jack_[0]);
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
    resize_same_as(rav1, jack_[0]);
    resize_same_as(rav2, obs2.jack_[0]);
    if (jack_.size() != obs2.jack_.size())
      boost::throw_exception(std::runtime_error("unequal number of bins in calculation of covariance matrix"));
    uint32_t k = jack_.size()-1;

    rav1 = 0;
    rav2 = 0;
    rav1 = std::accumulate(jack_.begin()+1, jack_.end(), rav1);
    rav2 = std::accumulate(obs2.jack_.begin()+1, obs2.jack_.end(), rav2);
    rav1 /= count_type(k);
    rav2 /= count_type(k);

    covariance_type cov = numeric::outer_product(jack_[1],obs2.jack_[1]);
    for (uint32_t i = 2; i < jack_.size(); ++i)
      cov += numeric::outer_product(jack_[i],obs2.jack_[i]);

    cov/=count_type(k);
    cov-= numeric::outer_product(rav1, rav2);
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
  return  discardedmeas_;
}

template <class T>
void SimpleObservableData<T>::set_thermalization(uint32_t thermal)
{
  if (nonlinear_operations_)
    boost::throw_exception(std::runtime_error("cannot set thermalization after nonlinear operations"));
  if (!can_set_thermalization())
    boost::throw_exception(std::runtime_error("cannot set thermalization"));
  if (binsize_) {
    discardedmeas_ = thermal ;
    discardedbins_ = (discardedmeas_ + binsize_ - 1) / binsize_;
    changed_ = true;
    valid_ = false;
    jack_valid_ = false;
  }
}

template <class T>
void SimpleObservableData<T>::collect_bins(std::size_t howmany)
{
  if (nonlinear_operations_)
    boost::throw_exception(std::runtime_error("cannot change bins after nonlinear operations"));
  if (values_.empty() || howmany <= 1) return;

  std::size_t newbins = values_.size() / howmany;

  // fill bins
  for (std::size_t i = 0; i < newbins; ++i) {
    values_[i] = values_[howmany * i];
    if (!values2_.empty()) values2_[i] = values2_[howmany * i];
    for (std::size_t j = 1; j < howmany; ++j) {
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
void SimpleObservableData<T>::set_bin_number(std::size_t binnum)
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

} // end namespace alps

#endif // ALPS_ALEA_SIMPLEOBSDATA_H
