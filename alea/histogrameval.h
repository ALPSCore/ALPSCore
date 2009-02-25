/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@comp-phys.org>,
*                            Fabian Stoeckli <fabstoec@phys.ethz.ch>,
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

#ifndef ALPS_ALEA_HISTOGRAMEVAL_H
#define ALPS_ALEA_HISTOGRAMEVAL_H

#include <alps/config.h>
#include <alps/alea/histogram.h>
#include <alps/alea/histogramdata.h>
#include <alps/parser/parser.h>

#include <algorithm>
#include <boost/functional.hpp>

#include <iostream>

//======================================================
//HistogramObservableEvaluator
//
//Observable class for Histograms
//======================================================

namespace alps {

template <class T>
class HistogramObservableEvaluator : public HistogramObservable<T>
{
  typedef uint32_t integer_type;
  typedef HistogramObservable<T> supertype;
public:
  template <class X>
  friend class HistogramObservableEvaluator;
  typedef typename supertype::value_type value_type;
  typedef typename supertype::range_type range_type;
  typedef typename supertype::count_type count_type;
  typedef typename std::vector<HistogramObservableData<T> >::iterator iterator;
  typedef typename std::vector<HistogramObservableData<T> >::const_iterator const_iterator;

  BOOST_STATIC_CONSTANT(uint32_t, version = type_traits<T>::type_tag
    + (type_traits<integer_type>::type_tag << 8) + (6<<16));

  //constructors
  HistogramObservableEvaluator(const std::string& n="");
  HistogramObservableEvaluator(const char* n);
  HistogramObservableEvaluator(const HistogramObservableEvaluator& eval);
  HistogramObservableEvaluator(const Observable& obs,const std::string&);
  HistogramObservableEvaluator(const Observable&);
  HistogramObservableEvaluator(const std::string& n, std::istream&, const XMLTag&);
  virtual ~HistogramObservableEvaluator() {}

  //assigning an observable
  const HistogramObservableEvaluator<T>& operator=(const HistogramObservableEvaluator<T>& eval);
  const HistogramObservableEvaluator<T>& operator=(const HistogramObservable<T>& obs);

  //add an observable
  HistogramObservableEvaluator<T>& operator<<(const HistogramObservable<T>& obs) {
    merge(obs); return *this;
  }

  void rename(const std::string& n) {
    Observable::rename(n);
    automatic_naming_ = false;
  }

  void rename(const std::string n, bool a) {
    Observable::rename(n);
    automatic_naming_=a;
  }
  ALPS_DUMMY_VOID reset(bool=false);

  value_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const {collect(); return all_.max BOOST_PREVENT_MACRO_SUBSTITUTION (); }
  value_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const {collect(); return all_.min BOOST_PREVENT_MACRO_SUBSTITUTION (); }
  value_type operator[](int i) const { collect(); return all_[i]; }

  count_type count() const {collect(); return all_.count(); }

  Observable* clone() const {
    HistogramObservableEvaluator<T>* my_eval = new HistogramObservableEvaluator<T>(*this);
    return my_eval;
  }

  void set_thermalization(uint32_t todiscard);
  uint32_t get_thermalization() const { collect(); return all_.get_thermalization(); }
  bool can_set_thermalization() const { collect(); return all_.can_set_thermalization(); }

  uint32_t number_of_runs() const;
  Observable* get_run(uint32_t) const;

  ALPS_DUMMY_VOID compact();

  ALPS_DUMMY_VOID output(std::ostream&) const;
  void output_histogram(std::ostream&) const;

  void operator<<(const HistogramObservableData<T>& obs);

#ifndef ALPS_WITHOUT_OSIRIS
  virtual uint32_t version_id() const { return version; }
  virtual void save(ODump& dump) const;
  virtual void load(IDump& dump);
#endif

  void merge(const Observable&);
  bool can_merge() const { return true; }
  bool can_merge(const Observable&) const;
  Observable* convert_mergeable() const { return clone(); }
  HistogramObservableEvaluator<T> make_evaluator() const { return *this; }

private:
  void collect() const;

  //?   mutable bool valid_;
  bool automatic_naming_;
  std::vector<HistogramObservableData<T> > runs_;
  mutable HistogramObservableData<T> all_;
};

typedef HistogramObservableEvaluator<int32_t> IntHistogramObsevaluator;
typedef HistogramObservableEvaluator<double> RealHistogramObsevaluator;

#ifndef ALPS_WITHOUT_OSIRIS

template <class T>
inline void HistogramObservableEvaluator<T>::save(ODump& dump) const
{
  Observable::save(dump);
  dump << runs_ << all_;
}

template <class T>
inline void HistogramObservableEvaluator<T>::load(IDump& dump)
{
  Observable::load(dump);
  dump >> runs_ >> all_;
}

#endif

template<class T>
inline void HistogramObservableEvaluator<T>::collect() const
{
  all_.collect_from(runs_);
  supertype::count_=all_.count();
  supertype::histogram_.resize(all_.size());
  for (std::size_t i=0;i<all_.size();++i)
    supertype::histogram_[i] = all_[i];
  supertype::thermalized_ = (supertype::count_ > 0);
}

template <class T>
inline const HistogramObservableEvaluator<T>& HistogramObservableEvaluator<T>::operator=(const HistogramObservableEvaluator<T>& eval)
{
  runs_ = eval.runs_;
  all_ = eval.all_;
  if (automatic_naming_ && supertype::name() == "") Observable::rename(eval.name());
  return *this;
}

template <class T>
inline const HistogramObservableEvaluator<T>&  HistogramObservableEvaluator<T>::operator=(const HistogramObservable<T>& obs)
{
  std::string oldname = supertype::name();
  bool a = automatic_naming_;
  HistogramObservableEvaluator<T> eval(obs);
  *this = eval;
  if (!a) rename(oldname);
  return *this;
}

template <class T>
inline void HistogramObservableEvaluator<T>::set_thermalization(uint32_t todiscard)
{
  std::for_each(runs_.begin(), runs_.end(), boost::bind2nd(boost::mem_fun_ref(&HistogramObservableData<T>::set_thermalization), todiscard));
}

template <class T>
inline void HistogramObservableEvaluator<T>::operator<<(const HistogramObservableData<T>& b)
{
  runs_.push_back(b);
}

template <class T>
inline void HistogramObservableEvaluator<T>::merge(const Observable& o)
{
  if (automatic_naming_ && supertype::name()=="") Observable::rename(o.name());
  if (dynamic_cast<const HistogramObservable<T>*>(&o)!=0) {
    if (dynamic_cast<const HistogramObservable<T>&>(o).is_thermalized())
      (*this) << HistogramObservableData<T>(dynamic_cast<const HistogramObservable<T>&>(o));
  } else {
    const HistogramObservableEvaluator<T>& eval =
      dynamic_cast<const HistogramObservableEvaluator<T>&>(o);
    if (automatic_naming_ && !eval.automatic_naming_) automatic_naming_ = false;
    for (unsigned int i = 0; i < eval.runs_.size(); ++i)
      (*this) << eval.runs_[i];
  }
  this->collect();
}


template <class T>
inline uint32_t HistogramObservableEvaluator<T>::number_of_runs() const
{
  return runs_.size();
}

template <class T>
inline Observable* HistogramObservableEvaluator<T>::get_run(uint32_t i) const
{
  HistogramObservableEvaluator<T>* res = new HistogramObservableEvaluator<T>(supertype::name());
  (*res) << runs_[i];
  return res;
}

template <class T>
inline ALPS_DUMMY_VOID HistogramObservableEvaluator<T>::reset(bool)
{
  runs_.clear();
  all_ = HistogramObservableData<T>();
  ALPS_RETURN_VOID
}

template <class T>
inline ALPS_DUMMY_VOID HistogramObservableEvaluator<T>::compact()
{
  collect();
  std::for_each(runs_.begin(), runs_.end(), boost::mem_fun_ref(&HistogramObservableData<T>::compact));
  all_.compact();
  ALPS_RETURN_VOID
}

template <class T>
ALPS_DUMMY_VOID HistogramObservableEvaluator<T>::output(std::ostream& out) const
{
  HistogramObservableEvaluator<T>::output_histogram(out);
  ALPS_RETURN_VOID
}

template<class T>
void HistogramObservableEvaluator<T>::output_histogram(std::ostream& out) const
{
  out << supertype::name() << ":\n";
  if(count()==0)
    out << " no measurements.\n";
  else
    for(integer_type j=0; j<supertype::histogram_.size();++j)
          out << " " <<j<<": "<<supertype::histogram_[j]<<std::endl;
}

template <class T>
inline bool HistogramObservableEvaluator<T>::can_merge(const Observable& obs) const
{
  return dynamic_cast<const HistogramObservable<T>*>(&obs) != 0;
}

template <class T>
inline HistogramObservableEvaluator<T>::HistogramObservableEvaluator(const std::string& n)
  : HistogramObservable<T>(n), automatic_naming_(n=="") {}

template <class T>
inline HistogramObservableEvaluator<T>::HistogramObservableEvaluator(const char* n)
  : HistogramObservable<T>(std::string(n)), automatic_naming_(false) {}

template <class T>
inline HistogramObservableEvaluator<T>::HistogramObservableEvaluator(const HistogramObservableEvaluator& eval)
  : HistogramObservable<T>(eval.name()), automatic_naming_(true), runs_(eval.runs_), all_(eval.all_){}

template <class T>
inline HistogramObservableEvaluator<T>::HistogramObservableEvaluator(const Observable& b, const std::string& n)
  : HistogramObservable<T>(n=="" ? b.name() : n),
    automatic_naming_(n=="")
{
  merge(b);
}

template <class T>
inline HistogramObservableEvaluator<T>::HistogramObservableEvaluator(const Observable& b)
  : HistogramObservable<T>(b.name()), automatic_naming_(true)
{
  if (dynamic_cast<const HistogramObservable<T>*>(&b)==0)
    merge(b);
  else
    (*this) = dynamic_cast<const HistogramObservable<T>&>(b).make_evaluator();
}

template <class T>
inline HistogramObservableEvaluator<T>::HistogramObservableEvaluator(const std::string& n, std::istream& infile, const XMLTag& intag)
  : HistogramObservable<T>(n),
    automatic_naming_(false),
    all_(infile,intag)
{}

} //end namespace alps

#endif


