/***************************************************************************
* ALPS++/alea library
*
* $Id$
*
* Copyright (C) 1997-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

#ifndef ALPS_ALEA_HISTOGRAM_H
#define ALPS_ALEA_HISTOGRAM_H

#include <alps/config.h>
#include <alps/alea/observable.h>
#include <alps/typetraits.h>

#include <vector>

#ifndef ALPS_WITHOUT_OSIRIS
# include <alps/osiris.h>
#endif

namespace alps {

template <class T, class INT = uint32_t>
class HistogramObservable : public Observable
{
public:
  enum { version=TypeTraits<T>::type_tag+(TypeTraits<INT>::type_tag << 8) + (2<<16)};
  HistogramObservable(const std::string& n);
  HistogramObservable(const std::string& n, T min, T max, T stepsize=1);
  void set_range(T min, T max, T stepsize=1);
  virtual Observable* clone() const {return new HistogramObservable<T,INT>(*this);}
  virtual ALPS_DUMMY_VOID reset(bool forthermalization=false);
  virtual ALPS_DUMMY_VOID output(std::ostream&) const;

#ifndef ALPS_WITHOUT_OSIRIS
  virtual uint32_t version_id() const { return version;}
  virtual void save(ODump& dump) const;
  virtual void load(IDump& dump);
#endif

  // thermalization support
  // virtual void set_thermalization(uint32_t todiscard) { ??? }
  virtual void set_thermalization(uint32_t) {}
  virtual uint32_t get_thermalization() const {return thermalcount_;}

  /** add a simple T-value to the Observable */
  void add(const T& x); // { b_.add(x); }
  /** add a simple T-value to the Observable */
  void operator<<(const T& x) { add(x); }

  // forward a few things from container
  
  typedef INT value_type;
  typedef typename std::vector<INT>::const_iterator const_iterator;
  typedef typename std::vector<INT>::const_reverse_iterator const_reverse_iterator;
  typedef typename std::vector<INT>::size_type size_type;
  const_iterator begin() const { return histogram_.begin();}
  const_iterator rbegin() const { return histogram_.rbegin();}
  const_iterator end() const { return histogram_.end();}
  const_iterator rend() const { return histogram_.rend();}
  size_type size() const { return histogram_.size();}
  value_type operator[](size_type i) const { return histogram_[i];}
  value_type at(size_type i) const { return histogram_.at(i);}

  value_type& operator[](size_type i) { return histogram_[i]; }
private:
  uint32_t thermalcount_;
  uint32_t count_;
  T min_;
  T max_;
  T stepsize_;
  std::vector<INT> histogram_;
};


template <class T, class INT>
HistogramObservable<T,INT>::HistogramObservable(const std::string& n) 
 : Observable(n),
   thermalcount_(0),
   count_(0),
   min_(std::numeric_limits<T>::max()),
   max_(std::numeric_limits<T>::min()),
   stepsize_(0)
 {
 }

template <class T, class INT>
inline HistogramObservable<T,INT>::HistogramObservable(const std::string& n, T min, T max, T stepsize) 
 : Observable(n),
   thermalcount_(0),
   count_(0)
{
  set_range(min,max,stepsize);  
}
 
template <class T, class INT>
inline void HistogramObservable<T,INT>::set_range(T min, T max, T stepsize) 
{
  if (count_!=0)
    boost::throw_exception(std::runtime_error("cannot change range of HistogramObservable after performing measurements"));
  min_=min;
  max_=max;
  stepsize_=stepsize;
  histogram_.resize(((std::numeric_limits<T>::is_integer ? max_+1 : max)-min)/stepsize);
}

template <class T, class INT>
inline void HistogramObservable<T,INT>::add(const T& x)
{
  if (x>= min_ && (std::numeric_limits<T>::is_integer ? x<= max_ : x<max_))
  {
    histogram_[(x-min_)/stepsize_]++;
    count_++;
  }
}

template <class T, class INT>
inline ALPS_DUMMY_VOID
HistogramObservable<T,INT>::reset(bool forthermalization)
{
  thermalcount_ = (forthermalization ? count_ : 0);
  count_=0;
  std::fill(histogram_.begin(),histogram_.end(),0);
  ALPS_RETURN_VOID
}

template <class T, class INT>
inline ALPS_DUMMY_VOID
HistogramObservable<T,INT>::output(std::ostream& out) const
{
  out << name() << ":\n";
  for (std::size_t i=0;i<histogram_.size();++i)
  {
    if(stepsize_==1)
      out << i+min_;
    else
      if (std::numeric_limits<T>::is_integer)
        out << "[" << min_+i*stepsize_ << "," << min_+(i+1)*stepsize_-1 << "]";
      else
        out << "[" << min_+i*stepsize_ << "," << min_+(i+1)*stepsize_ << "[";
    out << ": " << histogram_[i] << " entries.\n";
  }
  ALPS_RETURN_VOID
}

#ifndef ALPS_WITHOUT_OSIRIS

template <class T, class INT>
inline void HistogramObservable<T,INT>::save(ODump& dump) const
{
  Observable::save(dump);
  dump << thermalcount_ << count_ << min_ << max_ << stepsize_ << histogram_;
}

template <class T, class INT>
inline void HistogramObservable<T,INT>::load(IDump& dump)
{
  Observable::load(dump);
  dump >> thermalcount_ >> count_ >> min_ >> max_ >> stepsize_ >> histogram_;
}

#endif

} // end namespace alps

#endif // ALPS_ALEA_HISTOGRAM_H
