/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_ALEA_SIMPLEOBSERVABLE_H
#define ALPS_ALEA_SIMPLEOBSERVABLE_H

#include <alps/alea/abstractsimpleobservable.h>
#include <alps/alea/recordableobservable.h>
#include <alps/type_traits/change_value_type.hpp>
#include <alps/type_traits/is_scalar.hpp>
#include <alps/alea/type_tag.hpp>
#include <alps/utilities/encode.hpp>

#include <alps/hdf5.hpp>

namespace alps {

//=======================================================================
// SimpleObservable
//
// Observable class with variable autocorrelation analysis and binning
//-----------------------------------------------------------------------

template <class T,class BINNING>
class ALPS_TEMPL_DECL SimpleObservable: public AbstractSimpleObservable<T>, public RecordableObservable<T>
{
public:
  typedef typename AbstractSimpleObservable<T>::value_type value_type;
  typedef typename AbstractSimpleObservable<T>::time_type time_type;
  typedef typename AbstractSimpleObservable<T>::count_type count_type;
  typedef typename AbstractSimpleObservable<T>::result_type result_type;
  typedef typename AbstractSimpleObservable<T>::slice_index slice_index;
  typedef typename AbstractSimpleObservable<T>::label_type label_type;
  typedef typename change_value_type<T,int>::type convergence_type;
  typedef BINNING binning_type;

  BOOST_STATIC_CONSTANT(int,version=(type_tag<T>::value+ (binning_type::magic_id << 16)));
  /// the constructor needs a name and optionally specifications for the binning strategy
  SimpleObservable(const std::string& name=std::string(), const label_type& l=label_type())
   : AbstractSimpleObservable<T>(name,l) {}

  SimpleObservable(const std::string& name,const binning_type& b, const label_type& l=label_type())
   : AbstractSimpleObservable<T>(name,l), b_(b) {}

  SimpleObservable(const std::string& name,uint32_t s, const label_type& l=label_type())
   : AbstractSimpleObservable<T>(name,l), b_(s) {}

  SimpleObservable(const std::string& name,uint32_t s,uint32_t a, const label_type& l=label_type())
   : AbstractSimpleObservable<T>(name,l), b_(s, a) {}

  uint32_t version_id() const { return version;}

  Observable* clone() const {return new SimpleObservable<T,BINNING>(*this);}

  void output(std::ostream&) const;

  ALPS_DUMMY_VOID reset(bool why)
  {
    b_.reset(why);
  }

  result_type mean() const {return b_.mean();}
  bool has_variance() const { return b_.has_variance();}
  result_type variance() const {return b_.variance();}
  result_type error() const {return b_.error();}
  result_type error(unsigned bin_used) const {return b_.error(bin_used);}
  convergence_type converged_errors() const {return b_.converged_errors();}
  count_type count() const {return b_.count();}
  bool has_tau() const { return b_.has_tau;}
  time_type tau() const  { return b_.tau();}
  std::string representation() const { return hdf5_name_encode(this->name()); } 
    
    
  void operator<<(const T& x) { 
    if (size(x) == 0)
      boost::throw_exception(std::runtime_error("Cannot save a measurement of size 0."));
    b_ << x;
  }


  // Additional binning member functions

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
    
  const std::vector<value_type>& bins() const {return b_.bins();} 

  virtual void save(ODump& dump) const;
  virtual void load(IDump& dump);
  void extract_timeseries(ODump& dump) const { b_.extract_timeseries(dump);}

  virtual void save(hdf5::archive &) const;
  virtual void load(hdf5::archive &);

  virtual std::string evaluation_method(Target t) const
  { return (t==Mean || t== Variance) ? std::string("simple") : b_.evaluation_method();}

private:
  Observable* convert_mergeable() const;
  void write_more_xml(oxstream& oxs, slice_index it) const;
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
void
SimpleObservable<T,BINNING>::output(std::ostream& o) const
{
  if(count()!=0)
  {
    o << this->name ();
    output_helper<typename is_scalar<T>::type>::output(b_,o,this->label());
  }
}
    
template <class T,class BINNING>
void SimpleObservable<T,BINNING>::write_more_xml(oxstream& oxs, slice_index it) const
{
  output_helper<typename is_scalar<T>::type>::write_more_xml(b_, oxs, it);
}

template <class T,class BINNING>
inline void SimpleObservable<T,BINNING>::save(ODump& dump) const
{
  AbstractSimpleObservable<T>::save(dump);
  dump << b_;
}

template <class T,class BINNING>
inline void SimpleObservable<T,BINNING>::load(IDump& dump)
{
  AbstractSimpleObservable<T>::load(dump);
  dump >> b_;
}


template <class T,class BINNING> 
hdf5::archive & operator<<(hdf5::archive & ar,  SimpleObservable<T,BINNING> const& obs);

template <class T,class BINNING> 
hdf5::archive & operator>>(hdf5::archive & ar,  SimpleObservable<T,BINNING>& obs);

} // end namespace alps

#endif // ALPS_ALEA_SIMPLEOBSERVABLE_H
