/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@itp.phys.ethz.ch>,
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

/* $Id$ */

#ifndef ALPS_ALEA_ABSTRACTSIMPLEOBSERVABLE_H
#define ALPS_ALEA_ABSTRACTSIMPLEOBSERVABLE_H

#include <alps/alea/observable.h>
#include <alps/alea/abstractbinning.h>
#include <alps/alea/recordableobservable.h>
#include <alps/alea/output_helper.h>
#include <alps/alea/convergence.hpp>
#include <alps/type_traits/is_scalar.hpp>
#include <alps/type_traits/change_value_type.hpp>
#include <alps/type_traits/average_type.hpp>
#include <alps/type_traits/slice.hpp>
#include <alps/utility/resize.hpp>
#include <alps/utility/size.hpp>
#include <alps/utility/assign.hpp>

#include <boost/config.hpp>
#include <boost/tuple/tuple.hpp>

namespace alps {

template <class T>
inline bool error_underflow(T mean, T error)
{
  return ((error!= 0. && mean != 0.)  && (std::abs(mean) * 10.
             *std::sqrt(std::numeric_limits<T>::epsilon()) > std::abs(error)));
}

template <class T>
inline bool error_underflow(std::complex<T> mean, std::complex<T> error)
{
  return false;
}

class RealVectorObsevaluatorXMLHandler;


//=======================================================================
// AbstractSimpleObservable
//
// Observable class interface
//-----------------------------------------------------------------------

template <class T> class ALPS_TEMPL_DECL SimpleObservableEvaluator;

template <class T>
class ALPS_TEMPL_DECL AbstractSimpleObservable: public Observable
{
public:
  friend class RealVectorObsevaluatorXMLHandler;

  /// the data type of the observable
  typedef T value_type;

  /// the data type of averages and errors
  typedef typename average_type<T>::type result_type;

  typedef typename alps::slice_index<result_type>::type slice_index;
  /// the count data type: an integral type
  // typedef std::size_t count_type;
  typedef uint64_t count_type;

  /// the data type for autocorrelation times
  typedef typename change_value_type<T,double>::type time_type;
  typedef typename change_value_type<T,int>::type convergence_type;
  typedef typename change_value_type_replace_valarray<value_type,std::string>::type label_type;

  AbstractSimpleObservable(const std::string& name="", const label_type& l=label_type())
   : Observable(name), label_(l) {}

  virtual ~AbstractSimpleObservable() {}

  /// the number of measurements
  virtual count_type count() const =0;

  /// the mean value
  virtual result_type mean() const =0;

  /// the variance
  virtual result_type variance() const { boost::throw_exception(std::logic_error("No variance provided in observable"));  return result_type();}

  /// the error
  virtual result_type error() const =0;
  virtual convergence_type converged_errors() const =0;

  /// is information about the minimum and maximum value available?
  virtual bool has_minmax() const { return false;}

  /// the minimum value
  virtual value_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const { boost::throw_exception(std::logic_error("No min provided in observable")); return value_type();}

  /// the maximum value
  virtual value_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const { boost::throw_exception(std::logic_error("No max provided in observable")); return value_type();}

  /// is autocorrelation information available ?
  virtual bool has_tau() const { return false;}

  /// the autocorrelation time, throws an exception if not available
  virtual time_type tau() const { boost::throw_exception(std::logic_error("No autocorrelation time provided in observable")); return time_type();}

  /// is variance  available ?
  virtual bool has_variance() const { return false;}

  //@name binning information
  /// the number of bins
  virtual count_type bin_number() const { return 0;}
  /// the number of bins
  virtual count_type max_bin_number() const { return 0;}
  /// the number of measurements per bin
  virtual count_type bin_size() const { return 0;}
  /// the value of a bin
  virtual const value_type& bin_value(count_type) const
  { boost::throw_exception(std::logic_error("bin_value called but no bins present")); return *(new value_type());}
  /// the number of bins with squared values
  virtual count_type bin_number2() const { return 0;}
  /// the squared value of a bin
  virtual const value_type& bin_value2(count_type) const
  { boost::throw_exception(std::logic_error("bin_value2 called but no bins present")); return *(new value_type());}

  //@name Slicing of observables
  /** slice the data type using a single argument.
      This can easily be extended when needed to more data types.
      @param s the slice
      @param newname optionally a new name for the slice. Default is the
                     same name as the original observable
      */
  template <class S>
  SimpleObservableEvaluator<typename element_type<T>::type>
    slice(S s, const std::string& newname="") const;

  template <class S>
  SimpleObservableEvaluator<typename element_type<T>::type>
    operator[](S s) const { return slice(s);}

  void extract_timeseries(ODump& dump) const;

  void write_xml(oxstream&, const boost::filesystem::path& = boost::filesystem::path()) const;
  void write_xml_scalar(oxstream&, const boost::filesystem::path&) const;
  void write_xml_vector(oxstream&, const boost::filesystem::path&) const;

  virtual std::string evaluation_method(Target) const { return "";}

  operator SimpleObservableEvaluator<value_type> () const { return make_evaluator();}

  void set_label(const label_type& l) { label_=l;}
  const label_type& label() const { return label_;}

  void save(ODump& dump) const
  {
    Observable::save(dump);
    dump << label_;
  }

  void load(IDump& dump)
  {
    Observable::load(dump);
    if (dump.version() >= 303 || dump.version()==0)
      dump >> label_;
  }
  
void save(hdf5::archive &) const;
void load(hdf5::archive &);

private:
  virtual SimpleObservableEvaluator<value_type> make_evaluator() const
  {
    return SimpleObservableEvaluator<value_type>(*this,name());
  }
  friend class SimpleObservableEvaluator<value_type>;

  virtual void write_more_xml(oxstream&, slice_index = slice_index()) const {}

  label_type label_;
};


//=======================================================================
// Implementations
//=======================================================================

template <class T>
void AbstractSimpleObservable<T>::write_xml(oxstream& oxs, const boost::filesystem::path& fn_hdf5) const
{
  output_helper<typename is_scalar<T>::type>::write_xml(*this, oxs, fn_hdf5);
}

template <class T>
void AbstractSimpleObservable<T>::write_xml_scalar(oxstream& oxs, const boost::filesystem::path&) const {
  if (count())
  {
    std::string mm = evaluation_method(Mean);
    std::string em = evaluation_method(Error);
    std::string vm = evaluation_method(Variance);
    std::string tm = evaluation_method(Tau);

    oxs << start_tag("SCALAR_AVERAGE") << attribute("name", name());
    if (is_signed())
      oxs << attribute("signed","true");

    oxs << start_tag("COUNT") << no_linebreak << count() << end_tag("COUNT");

    oxs << start_tag("MEAN") << no_linebreak;
    if (mm != "")
      oxs << attribute("method", mm);

    int prec=(int)(4-slice_value(std::log10(std::abs(error()/mean())), 0));
    prec = (prec>=3 && prec<20 ? prec : 8);
    oxs << precision(slice_value(mean(), 0),prec) << end_tag("MEAN");

    oxs << start_tag("ERROR") << attribute("converged", convergence_to_text(converged_errors())) ;
    if (error_underflow(slice_value(mean(), 0),slice_value(error(), 0)))
      oxs << attribute("underflow","true");
    if (em != "")
      oxs << attribute("method", em);
    oxs << no_linebreak;
    oxs << precision(slice_value(error(), 0), 3) << end_tag("ERROR");

    if (has_variance()) {
      oxs << start_tag("VARIANCE") << no_linebreak;
      if (vm != "") oxs << attribute("method", vm);
      oxs << precision(slice_value(variance(), 0), 3) << end_tag("VARIANCE");
    }
    if (has_tau()) {
      oxs << start_tag("AUTOCORR") << no_linebreak;
      if (tm != "") oxs << attribute("method", tm);
      oxs << precision(slice_value(tau(), 0), 3) << end_tag("AUTOCORR");
    }
    write_more_xml(oxs);
    oxs << end_tag("SCALAR_AVERAGE");
  }
}

template <class T>
void AbstractSimpleObservable<T>::write_xml_vector(oxstream& oxs, boost::filesystem::path const &) const {
  if(count())
  {
    std::string mm = evaluation_method(Mean);
    std::string em = evaluation_method(Error);
    std::string vm = evaluation_method(Variance);
    std::string tm = evaluation_method(Tau);
    result_type mean_(mean());
    result_type error_(error());
    convergence_type conv_(converged_errors());
    result_type variance_;
    result_type tau_;
    if(has_tau())
    {
      resize_same_as(tau_,mean_);
      assign(tau_,tau());
    }
    if(has_variance())
    {
      resize_same_as(variance_,mean_);
      assign(variance_,variance());
    }

    oxs << start_tag("VECTOR_AVERAGE")<< attribute("name", name())
        << attribute("nvalues", alps::size(mean()));
    if (is_signed())
      oxs << attribute("signed","true");

    slice_index it,end;
    boost::tie(it,end) = slices(mean_);
    typename alps::slice_index<label_type>::type it2=slices(label()).first;
    while (it!=end)
    {
      std::string lab(slice_value(label(), it2));
      if (lab=="")
        lab=slice_name(mean_,it);
      oxs << start_tag("SCALAR_AVERAGE")
          << attribute("indexvalue",lab);
      oxs << start_tag("COUNT");
      oxs << no_linebreak;
      oxs << count();
      oxs << end_tag("COUNT");
      int prec=(count()==1) ? 19 : (int)(4-std::log10(std::abs(slice_value(error_,it)/slice_value(mean_,it))));
      prec = (prec>=3 && prec<20 ? prec : 8);
      oxs << start_tag("MEAN") << no_linebreak;
      if (mm != "") oxs << attribute("method", mm);
      oxs << precision(slice_value(mean_, it), prec)
          << end_tag("MEAN");

      oxs << start_tag("ERROR") << attribute("converged", convergence_to_text(slice_value(conv_,it))) << no_linebreak;
      if (error_underflow(slice_value(mean_, it),slice_value(error_, it)))
        oxs << attribute("underflow","true");
      if (em != "") oxs << attribute("method", em);
      oxs << precision(slice_value(error_, it), 3)
          << end_tag("ERROR");

      if (has_variance()) {
        oxs << start_tag("VARIANCE") << no_linebreak;
        if (vm != "") oxs << attribute("method", vm);
        oxs << precision(slice_value(variance_, it), 3)
            << end_tag("VARIANCE");
      }
      if (has_tau()) {
        oxs << start_tag("AUTOCORR") << no_linebreak;
        if (tm != "") oxs << attribute("method", tm);
        oxs << precision(slice_value(tau_, it), 3)
            << end_tag("AUTOCORR");
      }
      write_more_xml(oxs,it);
      ++it;
      ++it2;
      oxs << end_tag("SCALAR_AVERAGE");
    }
    oxs << end_tag("VECTOR_AVERAGE");
  }
}

} // end namespace alps


#include <alps/alea/simpleobseval.h>

namespace alps {

template <class T> template <class S>
inline SimpleObservableEvaluator<typename element_type<T>::type>
AbstractSimpleObservable<T>::slice (S s, const std::string& n) const
{
  if (dynamic_cast<const SimpleObservableEvaluator<T>*>(this)!=0)
    return dynamic_cast<const SimpleObservableEvaluator<T>*>(this)->slice(s,n);
  else
    return SimpleObservableEvaluator<T>(*this).slice(s,n);
}

} // end namespace alps


#endif // ALPS_ALEA_ABSTRACTSIMPLEOBSERVABLE_H
