/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2010 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_ALEA_HISTOGRAM_H
#define ALPS_ALEA_HISTOGRAM_H

#include <alps/config.h>
#include <alps/alea/observable.h>
#include <alps/alea/recordableobservable.h>
#include <alps/alea/output_helper.h>
#include <alps/osiris/dump.h>
#include <alps/osiris/std/vector.h>

#include <alps/ngs/mchdf5.hpp>
#include <alps/ngs/mchdf5/vector.hpp>

#include <vector>


namespace alps {

template <class T> class HistogramObservableEvaluator;
class RealHistogramObservableXMLHandler;

template <class T>
class HistogramObservable : public Observable, public RecordableObservable<T>
{
  typedef uint32_t integer_type;
public:
  friend class RealHistogramObservableXMLHandler;
  typedef integer_type value_type;
  typedef T range_type;
  typedef uint64_t count_type;
  typedef typename std::vector<integer_type>::const_iterator const_iterator;
  typedef typename std::vector<integer_type>::const_reverse_iterator const_reverse_iterator;
  typedef typename std::vector<integer_type>::size_type size_type;

  BOOST_STATIC_CONSTANT(uint32_t, version = alps::type_tag<T>::type::value
      + (alps::type_tag<integer_type>::value << 8) + (2<<16));

  HistogramObservable(const std::string& n="");
  HistogramObservable(const std::string& n, T min, T max, T stepsize=1);

  void set_range(T min, T max, T stepsize=1);
  virtual Observable* clone() const {return new HistogramObservable<T>(*this);}
  virtual ALPS_DUMMY_VOID reset(bool forthermalization=false);
  virtual ALPS_DUMMY_VOID output(std::ostream&) const;

  virtual uint32_t version_id() const { return version; }
  virtual void save(ODump& dump) const;
  virtual void load(IDump& dump);

  /** add a simple T-value to the Observable */
  void add(const T& x); // { b_.add(x); }
  /** add a simple T-value to the Observable */
  void operator<<(const T& x) { add(x); }

  // forward a few things from container

  const_iterator begin() const { return histogram_.begin();}
  const_iterator rbegin() const { return histogram_.rbegin();}
  const_iterator end() const { return histogram_.end();}
  const_iterator rend() const { return histogram_.rend();}
  size_type size() const { return histogram_.size();}
  value_type operator[](size_type i) const { return histogram_[i];}
  value_type at(size_type i) const { return histogram_.at(i);}


  bool can_merge() const {return false;}
  bool can_merge(const alps::Observable&) const {return false;}

  value_type& operator[](size_type i) { return histogram_[i]; }

  void write_xml(oxstream&, const boost::filesystem::path& = boost::filesystem::path()) const;


  inline count_type count() const {  return count_; }
  inline range_type stepsize() const {return stepsize_;}
  inline range_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const {return max_;}
  inline range_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const {return min_;}

  operator HistogramObservableEvaluator<T> () const { return make_evaluator();}

  void save(hdf5::archive &) const;
  void load(hdf5::archive &);

private:
  Observable* convert_mergeable() const;

  virtual HistogramObservableEvaluator<T> make_evaluator() const;

  friend class HistogramObservableEvaluator<T>;

  range_type min_;
  range_type max_;
  range_type stepsize_;

protected:
  mutable std::vector<value_type> histogram_;
  mutable count_type count_;
};


typedef HistogramObservable<int32_t> IntHistogramObservable;
typedef HistogramObservable<double> RealHistogramObservable;

template <class T>
inline Observable* HistogramObservable<T>::convert_mergeable() const
{
  HistogramObservableEvaluator<T>* my_eval= new HistogramObservableEvaluator<T>(*this);
  return my_eval;
}

template <class T>
HistogramObservable<T>::HistogramObservable(const std::string& n)
 : Observable(n),
   min_(std::numeric_limits<T>::max BOOST_PREVENT_MACRO_SUBSTITUTION ()),
   max_(std::numeric_limits<T>::min BOOST_PREVENT_MACRO_SUBSTITUTION ()),
   stepsize_(0),
   count_(0)
 {
 }

template <class T>
inline HistogramObservable<T>::HistogramObservable(const std::string& n, T min, T max, T stepsize)
 : Observable(n),
   count_(0)
{
  set_range(min,max,stepsize);
}


template <class T>
void HistogramObservable<T>::write_xml(oxstream& oxs, const boost::filesystem::path&) const
{
  if (count()) {
    oxs << start_tag("HISTOGRAM") << attribute("name",name())
        << attribute("nvalues",histogram_.size());
    for(unsigned int i=0;i<histogram_.size();++i) {
      oxs << start_tag("ENTRY") << attribute("indexvalue", i);
      oxs << start_tag("COUNT") << no_linebreak << count() <<end_tag("COUNT");
      oxs << start_tag("VALUE") << no_linebreak << histogram_[i] <<end_tag ("VALUE");
      oxs << end_tag("ENTRY");
    }
    oxs << end_tag("HISTOGRAM");
  }
}

template <class T>
inline void HistogramObservable<T>::set_range(T min, T max, T stepsize)
{
  if (count_!=0)
    boost::throw_exception(std::runtime_error("cannot change range of HistogramObservable after performing measurements"));
  min_=min;
  max_=max;
  stepsize_=stepsize;
  //std::cout<<"*** "<<(max-min)/stepsize<<std::endl;
  histogram_.resize(static_cast<size_type>((max-min)/stepsize+0.5));
}

template <class T>
inline void HistogramObservable<T>::add(const T& x)
{
  if (x>= min_ && x< max_)
  {
    //std::cout<<(x-min_)/stepsize_<<std::endl;
    histogram_[uint32_t((x-min_)/stepsize_)]++;
    count_++;
  }
}

template <class T>
inline ALPS_DUMMY_VOID
HistogramObservable<T>::reset(bool /* forthermalization */)
{
  count_=0;
  std::fill(histogram_.begin(),histogram_.end(),0);
  ALPS_RETURN_VOID
}

template <class T>
inline ALPS_DUMMY_VOID
HistogramObservable<T>::output(std::ostream& out) const
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

template <class T>
inline void HistogramObservable<T>::save(ODump& dump) const
{
  Observable::save(dump);
  dump << count_ << min_ << max_ << stepsize_ << histogram_;
}

template <class T>
inline void HistogramObservable<T>::load(IDump& dump)
{
  Observable::load(dump);
  bool thermalized_;
  uint32_t thermalcount_;

  if(dump.version() >= 306 || dump.version() == 0 /* version is not set */)
    dump >> count_ >> min_ >> max_ >> stepsize_ >> histogram_;
  else
    dump >> thermalized_ >> thermalcount_ >> count_ >> min_ >> max_ >> stepsize_ >> histogram_;
}

template <class T> inline void HistogramObservable<T>::save(hdf5::archive & ar) const {
    ar 
        << make_pvp("histogram", histogram_)
        << make_pvp("count",count_)
        << make_pvp("@min", min_)
        << make_pvp("@max", max_)
        << make_pvp("@stepsize", stepsize_)
    ;
}

template <class T> inline void HistogramObservable<T>::load(hdf5::archive & ar) {
    ar 
        >> make_pvp("histogram",histogram_)
        >> make_pvp("count",count_)
        >> make_pvp("@min", min_)
        >> make_pvp("@max", max_)
        >> make_pvp("@stepsize", stepsize_)
    ;
}

}

#include <alps/alea/histogrameval.h>

namespace alps {
template <class T>
HistogramObservableEvaluator<T> HistogramObservable<T>::make_evaluator() const
{
  return HistogramObservableEvaluator<T>(*this, name());
}

} // end namespace alps

#endif // ALPS_ALEA_HISTOGRAM_H
