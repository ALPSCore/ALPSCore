/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2004 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_ALEA_SIMPLEOBSERVABLE_H
#define ALPS_ALEA_SIMPLEOBSERVABLE_H

#include <alps/alea/abstractsimpleobservable.h>
#include <alps/alea/recordableobservable.h>

namespace alps {

//=======================================================================
// SimpleObservable
//
// Observable class with variable autocorrelation analysis and binning
//-----------------------------------------------------------------------

template <class T,class BINNING>
class SimpleObservable: public AbstractSimpleObservable<T>, public RecordableObservable<T>
{
  typedef AbstractSimpleObservable<T> super_type;
public:
  typedef typename AbstractSimpleObservable<T>::value_type value_type;
  typedef typename AbstractSimpleObservable<T>::time_type time_type;
  typedef typename AbstractSimpleObservable<T>::count_type count_type;
  typedef typename AbstractSimpleObservable<T>::result_type result_type;
  typedef typename AbstractSimpleObservable<T>::slice_iterator slice_iterator;
  typedef typename AbstractSimpleObservable<T>::label_type label_type;
  typedef typename obs_value_traits<T>::convergence_type convergence_type;
  typedef BINNING binning_type;

  BOOST_STATIC_CONSTANT(int,version=(obs_value_traits<T>::magic_id+ (binning_type::magic_id << 16)));
  /// the constructor needs a name and optionally specifications for the binning strategy
  SimpleObservable(const std::string& name=std::string(), const label_type& l=label_type())
   : AbstractSimpleObservable<T>(name,l) {}

  SimpleObservable(const std::string& name,const binning_type& b, const label_type& l=label_type())
   : AbstractSimpleObservable<T>(name,l), b_(b) {}

  SimpleObservable(const std::string& name,uint32_t s, const label_type& l=label_type())
   : AbstractSimpleObservable<T>(name,l), b_(s) {}

   
  uint32_t version_id() const { return version;}
  
  Observable* clone() const {return new SimpleObservable<T,BINNING>(*this);}

  ALPS_DUMMY_VOID output(std::ostream&) const;

  ALPS_DUMMY_VOID reset(bool forthermalization=false) 
  {
    b_.reset(forthermalization);
    ALPS_RETURN_VOID
  }

  result_type mean() const {return b_.mean();}
  bool has_variance() const { return b_.has_variance();}
  result_type variance() const {return b_.variance();}
  result_type error() const {return b_.error();}
  convergence_type converged_errors() const {return b_.converged_errors();}
  count_type count() const {return b_.count();}
  bool has_minmax() const { return b_.has_minmax();}
  value_type min() const {return b_.min();}
  value_type max() const {return b_.max();}
  bool has_tau() const { return b_.has_tau;}
  time_type tau() const  { return b_.tau();}
  
  virtual bool is_thermalized() const { return b_.is_thermalized();}
  uint32_t get_thermalization() const { return b_.get_thermalization();}
  bool can_set_thermalization() const { return b_.can_set_thermalization();}
  
  void operator<<(const T& x) { b_ << x;}
 

  //@{
  //@name Additional binning member functions
 
  count_type bin_size() const { return b_.bin_size();}
  /// resize bins to contain at least the given number of entries 
  void set_bin_size(count_type s) {b_.set_bin_size(s);}
  
  count_type bin_number() const { return b_.filled_bin_number();}
  count_type bin_number2() const { return b_.filled_bin_number2();}
  /// get the maximum number of bins
  count_type max_bin_number() const { return b_.max_bin_number();}
  /** set the maximum number of bins
      This will be the maximum number from now on if additional measurements are performed.
   */
  void set_bin_number(count_type n) {b_.set_bin_number(n);}
  const value_type& bin_value(count_type n) const {return b_.bin_value(n);}
  const value_type& bin_value2(count_type n) const {return b_.bin_value2(n);}
  
  //@}
    
#ifndef ALPS_WITHOUT_OSIRIS
  virtual void save(ODump& dump) const;
  virtual void load(IDump& dump);
  void extract_timeseries(ODump& dump) const { b_.extract_timeseries(dump);}
#endif

  ALPS_DUMMY_VOID compact() { b_.compact(); ALPS_RETURN_VOID }
  virtual std::string evaluation_method(Target t) const 
  { return (t==Mean || t== Variance) ? std::string("simple") : b_.evaluation_method();}

private:
  Observable* convert_mergeable() const;
  void write_more_xml(oxstream& oxs, slice_iterator it) const;
  binning_type b_;
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template <class T, class BINNING> const int SimpleObservable<T,BINNING>::version;
#endif


//=======================================================================
// Implementations
//=======================================================================

template <class T, class BINNING>
inline Observable* SimpleObservable<T,BINNING>::convert_mergeable() const
{
  return new SimpleObservableEvaluator<T>(*this);
}


template <class T,class BINNING>
ALPS_DUMMY_VOID
SimpleObservable<T,BINNING>::output(std::ostream& o) const 
{ 
  if(count()==0)
  {
    if(get_thermalization()>0)
    o << super_type::name() << " " << get_thermalization() << " thermalization steps, no measurements.\n";
  }
  else 
  {
    o << super_type::name ();
    output_helper<obs_value_traits<T>::array_valued>::output(b_,o,label());
  }
  ALPS_RETURN_VOID
}

template <class T,class BINNING>
void SimpleObservable<T,BINNING>::write_more_xml(oxstream& oxs, slice_iterator it) const 
{ 
  output_helper<obs_value_traits<T>::array_valued>::write_more_xml(b_, oxs, it);
}

#ifndef ALPS_WITHOUT_OSIRIS

template <class T,class BINNING>
inline void SimpleObservable<T,BINNING>::save(ODump& dump) const
{
  Observable::save(dump);
  dump << b_;
}

template <class T,class BINNING>
inline void SimpleObservable<T,BINNING>::load(IDump& dump) 
{
  Observable::load(dump);
  dump >> b_;
}

#endif

} // end namespace alps

#endif // ALPS_ALEA_SIMPLEOBSERVABLE_H
