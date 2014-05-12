/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2012 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>
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

#ifndef ALPS_ALEA_SIGNEDOBSERVABLE_H
#define ALPS_ALEA_SIGNEDOBSERVABLE_H

#include <alps/alea/simpleobseval.h>
#include <alps/type_traits/is_scalar.hpp>
#include <alps/type_traits/change_value_type.hpp>
#include <alps/type_traits/average_type.hpp>
#include <alps/numeric/round.hpp>
#include <alps/numeric/is_nonzero.hpp>
#include <typeinfo>

namespace alps {

//=======================================================================
// AbstractSignedObservable
//
// the base class for signed observables
//
//-----------------------------------------------------------------------

template <class OBS, class SIGN=double>
class AbstractSignedObservable
 : public AbstractSimpleObservable<typename OBS::value_type>
{
  typedef AbstractSimpleObservable<typename OBS::value_type> super_type;
public:
  typedef OBS observable_type;
  typedef SIGN sign_type;
  typedef typename observable_type::value_type value_type;
  typedef typename average_type<value_type>::type result_type;
  typedef AbstractSimpleObservable<value_type> base_type;
  typedef typename alps::slice_index<result_type>::type slice_index;
  // typedef std::size_t count_type;
  // *** we may need more than 32 Bit
  typedef uint64_t count_type;
  typedef typename change_value_type<value_type,double>::type time_type;
  typedef typename change_value_type<value_type,int>::type convergence_type;
  typedef typename super_type::label_type label_type;

  template <class X, class Y> friend class AbstractSignedObservable;

  BOOST_STATIC_CONSTANT(int,version=observable_type::version+(1<<24));

  AbstractSignedObservable(const OBS& obs, const std::string& s="Sign")
    : base_type(obs), obs_(obs), sign_name_(s), sign_(0)
  {  obs_.rename(s + " * "+super_type::name());}

  AbstractSignedObservable(const std::string& name="", const std::string& s="Sign", const label_type& l=label_type())
  : base_type(name,l), obs_(s +" * "+name), sign_name_(s), sign_(0) {}

  AbstractSignedObservable(const std::string& name, const char* s, const label_type& l=label_type())
  : base_type(name,l), obs_(std::string(s) +" * "+name), sign_name_(s), sign_(0) {}

  template <class OBS2>
  AbstractSignedObservable(const AbstractSignedObservable<OBS2,SIGN>& o)
  : base_type(o.name(),o.label()), obs_(o.obs_), sign_name_(o.sign_name_), sign_(o.sign_) {}

  template <class ARG>
  AbstractSignedObservable(const std::string& name,const ARG& arg, const label_type& l=label_type())
  : base_type(name,l), obs_("Sign * "+name,arg), sign_name_("Sign"), sign_(0) {}

  template <class ARG>
  AbstractSignedObservable(const std::string& name,std::string& s, const ARG& arg, const label_type& l=label_type())
   : base_type(name,l), obs_(s + " * "+name,arg), sign_name_(s), sign_(0) {}

  ~AbstractSignedObservable() {}

  uint32_t version_id() const { return version;}

  ALPS_DUMMY_VOID reset(bool forthermalization) { obs_.reset(forthermalization); ALPS_RETURN_VOID; }

  ALPS_DUMMY_VOID output(std::ostream& out) const
  {
    output_helper<typename alps::is_scalar<value_type>::type>::output(*this,out);
    obs_.output(out); ALPS_RETURN_VOID;
  }
  void output_scalar(std::ostream&) const;
  void output_vector(std::ostream&) const;
  void write_xml(oxstream& oxs, const boost::filesystem::path& p) const
  {
    base_type::write_xml(oxs,p);
    obs_.write_xml(oxs,p);
  }

  count_type count() const { return obs_.count();}
  result_type mean() const { return make_evaluator().mean();}
  result_type error() const { return make_evaluator().error();}
  convergence_type converged_errors() const { return make_evaluator().converged_errors();}

  SimpleObservableEvaluator<value_type> make_evaluator() const
  {
    SimpleObservableEvaluator<value_type> result(obs_);
    result.set_label(super_type::label());
    result /= static_cast<SimpleObservableEvaluator<sign_type> >(dynamic_cast<const AbstractSimpleObservable<sign_type>&>(sign()));
    result.rename(super_type::name());
    return result;
  }

  template <class S>
  AbstractSignedObservable<SimpleObservableEvaluator<typename element_type<value_type>::type>,SIGN>
    slice(S s, const std::string& newname="") const
  {
    AbstractSignedObservable<SimpleObservableEvaluator<typename element_type<value_type>::type>,SIGN> result(super_type::name());
    result.sign_=sign_;
    result.sign_name_=sign_name_;
    if(!newname.empty())
      result.rename(newname);
    result.obs_=obs_.slice(s);
    result.obs_.rename(sign_name_+" * " + super_type::name());
    return result;
  }

  template <class S>
  AbstractSignedObservable<SimpleObservableEvaluator<typename element_type<value_type>::type>,SIGN>
    operator[](S s) const { return slice(s);}

  void save(ODump& dump) const;
  void load(IDump& dump);

  void save(hdf5::archive &) const;
  void load(hdf5::archive &);

  Observable* clone() const { return new AbstractSignedObservable<OBS,SIGN>(*this);}

  bool is_signed() const { return true;}
  void set_sign_name(const std::string& signname) {sign_name_=signname;}
  void set_sign(const Observable& sign);
  void clear_sign() { sign_=0;}
  const Observable& sign() const;
  const std::string sign_name() const { return sign_name_;}
  const Observable& signed_observable() const { return obs_;}

  uint32_t number_of_runs() const { return obs_.number_of_runs();}
  Observable* get_run(uint32_t) const;

protected:
  Observable* convert_mergeable() const
  { return new AbstractSignedObservable<SimpleObservableEvaluator<value_type>,SIGN>(*this);}

  void write_more_xml(oxstream& oxs, slice_index it) const;

  void merge(const Observable& o) { obs_.merge(o.signed_observable());}
  bool can_merge() const { return obs_.can_merge();}
  bool can_merge(const Observable& o) const { return obs_.can_merge(o.signed_observable());}

  observable_type obs_;
  std::string sign_name_;
  const Observable* sign_;
};


  class A {};
//=======================================================================
// SignedObservable
//
// a signed observable that can record new measurements
//
//-----------------------------------------------------------------------

template <class OBS, class SIGN=double>
class SignedObservable
 : public AbstractSignedObservable<OBS,SIGN>,
   public RecordableObservable<typename OBS::value_type,SIGN>
{
  typedef AbstractSignedObservable<OBS,SIGN> super_type;
public:
  typedef OBS observable_type;
  typedef SIGN sign_type;
  typedef AbstractSignedObservable<OBS,SIGN> base_type;
  typedef typename observable_type::value_type value_type;
  typedef typename average_type<value_type>::type result_type;
  typedef std::size_t count_type;
  typedef typename change_value_type<value_type,double>::type time_type;
  typedef typename super_type::label_type label_type;

  SignedObservable(const OBS& obs, const std::string& s="Sign") : base_type(obs,s) {}
  SignedObservable(const std::string& name="", const std::string& s="Sign", const label_type& l=label_type())
  : base_type(name,s,l) {}
  SignedObservable(const std::string& name, const char * s, const label_type& l=label_type())
  : base_type(name,s,l) {}
  template <class ARG>
  SignedObservable(const std::string& name,const ARG& arg, const label_type& l=label_type())
  : base_type(name,arg,l) {}
  template <class ARG>
  SignedObservable(const std::string& name,std::string& s, const ARG& arg, const label_type& l=label_type())
  : base_type(name,s,arg,l) {}
  ~SignedObservable() {}

  Observable* clone() const { return new SignedObservable<OBS,SIGN>(*this);}

  void operator<<(const value_type& x) { super_type::obs_ << x;}
  void add(const value_type& x) { operator<<(x);}
  void add(const value_type& x, sign_type s) { add(x*static_cast<typename element_type<value_type>::type >(s));}
  void write_hdf5(const boost::filesystem::path& fn_hdf, std::size_t realization=0, std::size_t clone=0) const;
  void read_hdf5 (const boost::filesystem::path& fn_hdf, std::size_t realization=0, std::size_t clone=0);
};


//=======================================================================
// helper function
//
// make_observable creates a signed or non-signed observable
//-----------------------------------------------------------------------

template <class OBS>
boost::shared_ptr<Observable> make_observable(const OBS& obs, bool issigned=false)
{
  if (issigned)
    return boost::shared_ptr<Observable>(new SignedObservable<OBS,double>(obs));
  else
    return boost::shared_ptr<Observable>(obs.clone());
}

template <class OBS, class SIGN>
boost::shared_ptr<Observable>  make_observable(const OBS& obs, const std::string& s, SIGN, bool issigned=true)
{
  if (issigned)
    return boost::shared_ptr<Observable>(new SignedObservable<OBS,SIGN>(obs,s));
  else
    return boost::shared_ptr<Observable>(obs.clone());
}


//=======================================================================
// implementations
//-----------------------------------------------------------------------


template <class OBS, class SIGN>
const Observable& AbstractSignedObservable<OBS,SIGN>::sign() const
{
  if (!sign_)
    boost::throw_exception(std::logic_error("Sign requested but not set"));
  return *sign_;
}

template <class OBS, class SIGN>
void AbstractSignedObservable<OBS,SIGN>::set_sign(const Observable& sign)
{
  if (sign_name_.empty())
      sign_name_=sign.name();
  else
    if (sign_name_!=sign.name())
      boost::throw_exception(std::logic_error("Sign observable and sign name are inconsistent"));
  sign_=&sign;
}


template <class OBS, class SIGN>
void AbstractSignedObservable<OBS,SIGN>::save(ODump& dump) const
{
  AbstractSimpleObservable<value_type>::save(dump);
  obs_.save(dump);
  dump << sign_name_;
}

template <class OBS, class SIGN>
void AbstractSignedObservable<OBS,SIGN>::load(IDump& dump)
{
  AbstractSimpleObservable<value_type>::load(dump);
  obs_.load(dump);
  dump >> sign_name_;
  clear_sign();
}

template <class OBS, class SIGN> void AbstractSignedObservable<OBS,SIGN>::save(hdf5::archive & ar) const {
    super_type::save(ar);
    ar
        << make_pvp("@sign", sign_name_)
        << make_pvp("../" + obs_.name(), obs_)
    ;
}
template <class OBS, class SIGN>
void AbstractSignedObservable<OBS,SIGN>::load(hdf5::archive & ar) {
    super_type::load(ar);
    ar
        >> make_pvp("@sign", sign_name_)
    ;
    obs_.rename(sign_name_ + " * " + super_type::name());
    ar
        >> make_pvp("../" + obs_.name(), obs_)
    ;
    clear_sign();
}

template <class OBS, class SIGN>
void AbstractSignedObservable<OBS,SIGN>::write_more_xml(oxstream& oxs, slice_index) const
{
  oxs << start_tag("SIGN") << attribute("signed_observable",obs_.name());
  if (!sign_name_.empty())
     oxs << attribute("sign",sign_name_);
  oxs << end_tag("SIGN");
}

template <class OBS, class SIGN>
Observable* AbstractSignedObservable<OBS,SIGN>::get_run(uint32_t n) const
{
  AbstractSignedObservable* result = new AbstractSignedObservable(super_type::name());
  result->sign_=sign_;
  result->sign_name_=sign_name_;
  Observable* o = obs_.get_run(n);
  result->obs_ = dynamic_cast<OBS&>(*o);
  delete o;
  return result;
}

template <class OBS, class SIGN>
void AbstractSignedObservable<OBS,SIGN>::output_scalar(std::ostream& out) const
{
  out << super_type::name();
  if(count()==0)
    out << " no measurements.\n";
  else {
    out << ": " << alps::numeric::round<2>(mean()) << " +/- " << alps::numeric::round<2>(error());
    if (alps::numeric::is_nonzero<2>(error())) {
      if (!sign_name_.empty())
        out << "; sign in observable \"" << sign_name_ << "\"";
      if (converged_errors()==MAYBE_CONVERGED)
        out << " WARNING: check error convergence";
      if (converged_errors()==NOT_CONVERGED)
        out << " WARNING: ERRORS NOT CONVERGED!!!";
      if (error_underflow(mean(),error()))
        out << " Warning: potential error underflow. Errors might be smaller";
    }
    out << std::endl;
  }
}

template <class OBS, class SIGN>
void AbstractSignedObservable<OBS,SIGN>::output_vector(std::ostream& out) const
{
  out << super_type::name();
  if (!sign_name_.empty())
    out << "; sign in observable \"" << sign_name_ << "\"";

  if(count()==0)
    out << ": no measurements.\n";
  else {
    out << std::endl;
    result_type value_(mean());
    result_type error_(error());
    convergence_type conv_(converged_errors());
    typename alps::slice_index<label_type>::type it2=slices(this->label()).first;
    for (typename alps::slice_index<result_type>::type sit = slices(value_).first;
          sit!=slices(value_).second;++sit,++it2)
    {
      std::string lab=slice_value(super_type::label(),it2);
      if (lab=="")
        lab=slice_name(value_,sit);
      out << "Entry[" <<lab << "]: "
          << alps::numeric::round<2>(slice_value(value_,sit)) << " +/- "
          << alps::numeric::round<2>(slice_value(error_,sit));
      if (alps::numeric::is_nonzero<2>(slice_value(error_,sit))) {
        if (slice_value(conv_,sit)==MAYBE_CONVERGED)
          out << " WARNING: check error convergence";
        if (slice_value(conv_,sit)==NOT_CONVERGED)
          out << " WARNING: ERRORS NOT CONVERGED!!!";
        if (error_underflow(slice_value(value_,sit),slice_value(error_,sit)))
          out << " Warning: potential error underflow. Errors might be smaller";
      }
      out << std::endl;
    }
  }
}

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template <class OBS, class SIGN> const int AbstractSignedObservable<OBS,SIGN>::version;
#endif

}

#endif // ALPS_ALEA_SIGNEDOBSERVABLE_H
