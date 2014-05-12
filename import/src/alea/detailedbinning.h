/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2012 by Matthias Troyer <troyer@comp-phys.org>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@comp-phys.org>,
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

#ifndef ALPS_ALEA_DETAILEDBINNING_H
#define ALPS_ALEA_DETAILEDBINNING_H

#include <alps/config.h>
#include <alps/alea/observable.h>
#include <alps/alea/simpleobservable.h>
#include <alps/alea/simplebinning.h>
#include <alps/type_traits/change_value_type.hpp>
#include <alps/type_traits/average_type.hpp>
#include <boost/config.hpp>
#include <valarray>

//=======================================================================
// DetailedBinning
//
// detailed binning strategy
//-----------------------------------------------------------------------

namespace alps{

template <class T=double>
class BasicDetailedBinning : public SimpleBinning<T> {
public:
  typedef T value_type;  
  typedef typename change_value_type<T,double>::type time_type;
  typedef std::size_t size_type;
  typedef typename average_type<T>::type result_type;

  BOOST_STATIC_CONSTANT(bool, has_tau=true);
  BOOST_STATIC_CONSTANT(int, magic_id=3);

  BasicDetailedBinning(uint32_t binsize=1, uint32_t binnum=std::numeric_limits<uint32_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ());

  void reset(bool=false);
  void operator<<(const T& x);

  
  uint32_t max_bin_number() const { return maxbinnum_;}
  uint32_t bin_number() const;
  uint32_t filled_bin_number() const;
  uint32_t filled_bin_number2() const 
  { return(values2_.size() ? filled_bin_number() : 0);}
  
  void set_bin_number(uint32_t binnum);
  void collect_bins(uint32_t howmany);
  
  uint32_t bin_size() const { return binsize_;}
  void set_bin_size(uint32_t binsize);

  const value_type& bin_value(uint32_t i) const { return values_[i];}
  const value_type& bin_value2(uint32_t i) const { return values2_[i];}
    
  const std::vector<value_type>& bins() const { return values_;}  
  
  void compact();
  
  void save(ODump& dump) const;
  void load(IDump& dump);
  void extract_timeseries(ODump& dump) const { dump << binsize_ << values_.size() << binentries_ << values_;}

  void save(hdf5::archive &) const;
  void load(hdf5::archive &);

private:
  uint32_t binsize_;       // number of measurements per bin
  uint32_t minbinsize_;    // minimum number of measurements per bin
  uint32_t maxbinnum_;      // maximum number of bins 
  uint32_t  binentries_; // number of measurements in last bin
  std::vector<value_type> values_; // bin values
  std::vector<value_type> values2_; // bin values of squares
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template <class T> const bool BasicDetailedBinning<T>::has_tau;
#endif

template<class T> class DetailedBinning : public BasicDetailedBinning<T>
{
public:
  typedef T value_type;
  BOOST_STATIC_CONSTANT(int, magic_id=4);
  DetailedBinning(uint32_t binnum=128, uint32_t = 0) 
  : BasicDetailedBinning<T>(1,binnum==0 ? 128 : binnum) {}
};

template<class T> class FixedBinning : public BasicDetailedBinning<T>
{
public:
  typedef T value_type;
  BOOST_STATIC_CONSTANT(int, magic_id=5);
  FixedBinning(uint32_t binsize=1, uint32_t = 0)
  : BasicDetailedBinning<T>(binsize,std::numeric_limits<uint32_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ()) {}
};

typedef SimpleObservable<int32_t,DetailedBinning<int32_t> > IntObservable;
typedef SimpleObservable<double,DetailedBinning<double> > RealObservable;
typedef SimpleObservable<float,DetailedBinning<float> > FloatObservable;
typedef SimpleObservable<std::complex<double>,DetailedBinning<std::complex<double> > > ComplexObservable;
typedef SimpleObservable<double,FixedBinning<double> > RealTimeSeriesObservable;
typedef SimpleObservable<int32_t,FixedBinning<int32_t> > IntTimeSeriesObservable;

typedef SimpleObservable< std::valarray<int32_t> , 
                         DetailedBinning<std::valarray<int32_t> > > IntVectorObservable;
typedef SimpleObservable< std::valarray<double> , 
                         DetailedBinning<std::valarray<double> > > RealVectorObservable;
typedef SimpleObservable< std::valarray<float> , 
                         DetailedBinning<std::valarray<float> > > FloatVectorObservable;
//typedef SimpleObservable< std::valarray<std::complex<double> > , 
//                         DetailedBinning<std::valarray<std::complex<double> > > > ComplexVectorObservable;
typedef SimpleObservable< std::valarray<int32_t> , 
                         FixedBinning<std::valarray<int32_t> > > IntVectorTimeSeriesObservable;
typedef SimpleObservable< std::valarray<double> , 
                         FixedBinning<std::valarray<double> > > RealVectorTimeSeriesObservable;
//typedef SimpleObservable< std::valarray<std::complex<double> > , 
//                         FixedBinning<std::valarray<std::complex<double> > > > ComplexVectorTimeSeriesObservable;


template <class T>
inline BasicDetailedBinning<T>::BasicDetailedBinning(uint32_t binsize, uint32_t binnum)
 : SimpleBinning<T>(),
   binsize_(0), minbinsize_(binsize), maxbinnum_(binnum), binentries_(0)    
{
  reset();
}

template <class T>
inline void BasicDetailedBinning<T>::compact()
{
  values_.clear();
  values2_.clear();
  binsize_=minbinsize_;
  binentries_=0;
}

template <class T>
inline void BasicDetailedBinning<T>::reset(bool forthermal)
{
  compact();
  SimpleBinning<T>::reset(forthermal);
}

template <class T>
inline void BasicDetailedBinning<T>::operator<<(const T& x)
{
  if (values_.empty())
  { 
    // start first bin
    values_.push_back(x);
    values2_.push_back(x*x);
    binentries_ = 1;
    binsize_=1;
  }
  else if (values_.size()==1 && binentries_ < minbinsize_)
  {
    // fill first bin
    values_[0]+=x;
    values2_[0]+=x*x;
    binentries_++;
    binsize_++;
  }
  else if (binentries_==binsize_) // have a full bin
  {
    if(values_.size()<maxbinnum_)
    {
      // start a new bin
      values_.push_back(x);
      values2_.push_back(x*x);
      binentries_ = 1;
    }
    else
    {
      // halve the bins and add
      collect_bins(2);
      *this << x; // and just call again
      return;
    }
  }
  else
  {
    values_[values_.size()-1] += x;
    values2_[values_.size()-1] += x*x;
    ++binentries_;
  }
  SimpleBinning<T>::operator<<(x);
}

template <class T>
void BasicDetailedBinning<T>::collect_bins(uint32_t howmany)
{
  if (values_.empty() || howmany<=1)
    return;
    
  uint32_t newbins = (values_.size()+howmany-1)/howmany;
  
  // full bins
  for (uint32_t i=0;i<values_.size()/howmany;++i)
  {
    if(howmany*i !=i){
      values_[i]=values_[howmany*i];
      values2_[i]=values2_[howmany*i];
    }
    for (uint32_t j = 1 ; j<howmany;++j)
    {
      values_[i] += values_[howmany*i+j];
      values2_[i] += values2_[howmany*i+j];
    }
  }
  
  // last part of partly full bins
  values_[newbins-1]=values_[howmany*(newbins-1)];
  values2_[newbins-1]=values2_[howmany*(newbins-1)];
  for ( uint32_t i=howmany*(newbins-1)+1;i<values_.size();++i){
    values_[newbins-1]+=values_[i];
    values2_[newbins-1]+=values2_[i];
  }
    
  // how many in last bin?
  binentries_+=((values_.size()-1)%howmany)*binsize_;
  binsize_*=howmany;

  values_.resize(newbins);
  values2_.resize(newbins);
}

template <class T>
void BasicDetailedBinning<T>::set_bin_size(uint32_t minbinsize)
{
  minbinsize_=minbinsize;
  if(binsize_ < minbinsize_ && binsize_ > 0)
    collect_bins((minbinsize-1)/binsize_+1);
}

template <class T>
void BasicDetailedBinning<T>::set_bin_number(uint32_t binnum)
{
  maxbinnum_=binnum;
  if(values_.size() > maxbinnum_)
    collect_bins((values_.size()-1)/maxbinnum_+1);
}

template <class T>
inline uint32_t BasicDetailedBinning<T>::bin_number() const 
{ 
  return values_.size();
}

template <class T>
inline uint32_t BasicDetailedBinning<T>::filled_bin_number() const 
{ 
  if(values_.size()==0) return 0;
  else return values_.size() + (binentries_ ==binsize_ ? 0 : -1);
}

template <class T>
inline void BasicDetailedBinning<T>::save(ODump& dump) const
{
  SimpleBinning<T>::save(dump);
  dump << binsize_ << minbinsize_ << maxbinnum_<< binentries_ << values_
       << values2_;
}

template <class T>
inline void BasicDetailedBinning<T>::load(IDump& dump) 
{
  SimpleBinning<T>::load(dump);
  dump >> binsize_ >> minbinsize_ >> maxbinnum_ >> binentries_ >> values_
       >> values2_;
}

template <class T> inline void BasicDetailedBinning<T>::save(hdf5::archive & ar) const {
    SimpleBinning<T>::save(ar);
    if (values_.size() && values2_.size()) {
        ar
            << make_pvp("timeseries/partialbin", values_.back())
            << make_pvp("timeseries/partialbin/@count", binentries_)
            << make_pvp("timeseries/partialbin2", values2_.back())
            << make_pvp("timeseries/partialbin2/@count", binentries_)
        ;
        value_type value = values_.back();
        const_cast<BasicDetailedBinning<T> *>(this)->values_.pop_back();
        value_type value2 = values2_.back();
        const_cast<BasicDetailedBinning<T> *>(this)->values2_.pop_back();
        ar
            << make_pvp("timeseries/data", values_)
            << make_pvp("timeseries/data/@binningtype", "linear")
            << make_pvp("timeseries/data/@minbinsize", minbinsize_)
            << make_pvp("timeseries/data/@binsize", binsize_)
            << make_pvp("timeseries/data/@maxbinnum", maxbinnum_)
            << make_pvp("timeseries/data2", values2_)
            << make_pvp("timeseries/data2/@binningtype", "linear")
            << make_pvp("timeseries/data2/@minbinsize", minbinsize_)
            << make_pvp("timeseries/data2/@binsize", binsize_)
            << make_pvp("timeseries/data2/@maxbinnum", maxbinnum_)
        ;
        const_cast<BasicDetailedBinning<T> *>(this)->values_.push_back(value);
        const_cast<BasicDetailedBinning<T> *>(this)->values2_.push_back(value2);
    } else {
        ar
            << make_pvp("timeseries/data", values_)
            << make_pvp("timeseries/data/@binningtype", "linear")
            << make_pvp("timeseries/data/@minbinsize", minbinsize_)
            << make_pvp("timeseries/data/@binsize", binsize_)
            << make_pvp("timeseries/data/@maxbinnum", maxbinnum_)
            << make_pvp("timeseries/data2", values2_)
            << make_pvp("timeseries/data2/@binningtype", "linear")
            << make_pvp("timeseries/data2/@minbinsize", minbinsize_)
            << make_pvp("timeseries/data2/@binsize", binsize_)
            << make_pvp("timeseries/data2/@maxbinnum", maxbinnum_)
        ;
    }
}

template <class T> inline void BasicDetailedBinning<T>::load(hdf5::archive & ar) {
    SimpleBinning<T>::load(ar);
    ar 
        >> make_pvp("timeseries/data", values_)
        >> make_pvp("timeseries/data/@minbinsize", minbinsize_)
        >> make_pvp("timeseries/data/@binsize", binsize_)
        >> make_pvp("timeseries/data/@maxbinnum", maxbinnum_)
        >> make_pvp("timeseries/data2", values2_)
    ;
    if (ar.is_data("timeseries/partialbin")) {
        value_type value, value2;
        ar 
            >> make_pvp("timeseries/partialbin", value)
            >> make_pvp("timeseries/partialbin2", value2)
            >> make_pvp("timeseries/partialbin/@count", binentries_)
        ;
        values_.push_back(value);
        values2_.push_back(value2);
    }
}

} // end namespace alps

#endif // ALPS_ALEA_DETAILEDBINNING_H
