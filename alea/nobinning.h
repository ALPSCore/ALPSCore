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

#ifndef ALPS_ALEA_NOBINNING_H
#define ALPS_ALEA_NOBINNING_H

#include <alps/config.h>
#include <alps/alea/simpleobservable.h>
#include <alps/alea/abstractbinning.h>
#include <alps/alea/nan.h>
#include <alps/multi_array.hpp>

#ifdef ALPS_HAVE_VALARRAY
# include <valarray>
#endif

//=======================================================================
// NoBinning
//
// uncorrelated observable strategy
//-----------------------------------------------------------------------

namespace alps {

template <class T=double>
class NoBinning : public AbstractBinning<T>
{
  typedef AbstractBinning<T> super_type;
 public:
  typedef T value_type;
  typedef typename obs_value_traits<T>::size_type size_type;
  typedef typename obs_value_traits<T>::count_type count_type;
  typedef typename obs_value_traits<T>::result_type result_type;
  typedef typename obs_value_traits<T>::convergence_type convergence_type;
  
  BOOST_STATIC_CONSTANT(bool, has_tau=false);
  BOOST_STATIC_CONSTANT(int, magic_id=1);
  
  NoBinning(uint32_t=0);
  virtual ~NoBinning() {}
 
  void reset(bool=false);
  inline void operator<<(const value_type& x);
  
  result_type mean() const;
  result_type variance() const;
  result_type error() const;
  convergence_type converged_errors() const;

  uint32_t count() const { return super_type::is_thermalized() ? count_ : 0;}

  bool has_minmax() const { return true;}
  value_type min() const {return min_;}
  value_type max() const {return max_;}
  
  uint32_t get_thermalization() const { return super_type::is_thermalized() ? thermal_count_ : count_;}
    
  void output_scalar(std::ostream& out) const;
  template <class L> void output_vector(std::ostream& out, const L& l) const;
#ifndef ALPS_WITHOUT_OSIRIS
  virtual void save(ODump& dump) const;
  virtual void load(IDump& dump);
#endif

 private:
    value_type sum_;       // sum of measurements
    value_type sum2_;      // sum of squared measurements
    uint32_t count_;          // total number of measurements
    uint32_t thermal_count_; // measurements done for thermalization
    value_type min_,max_;  // minimum and maximum value
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template <class T> const bool NoBinning<T>::has_tau;
#endif

typedef SimpleObservable<int32_t,NoBinning<int32_t> > SimpleIntObservable;
typedef SimpleObservable<double,NoBinning<double> > SimpleRealObservable;
typedef SimpleObservable<std::complex<double>,NoBinning<std::complex<double> > > SimpleComplexObservable;
#ifdef ALPS_HAVE_VALARRAY
typedef SimpleObservable< std::valarray<int32_t> , NoBinning<std::valarray<int32_t> > > SimpleIntVectorObservable;
typedef SimpleObservable< std::valarray<double> , NoBinning<std::valarray<double> > > SimpleRealVectorObservable;
typedef SimpleObservable< std::valarray<std::complex<double> > , 
                         NoBinning<std::valarray<std::complex<double> > > > SimpleComplexVectorObservable;
#endif
typedef SimpleObservable< alps::multi_array<int32_t,2> , NoBinning<alps::multi_array<int32_t,2> > > SimpleInt2DArrayObservable;
typedef SimpleObservable< alps::multi_array<double,2> , NoBinning<alps::multi_array<double,2> > > SimpleReal2DArrayObservable;
typedef SimpleObservable< alps::multi_array<std::complex<double>,2> , NoBinning<alps::multi_array<std::complex<double>,2> > > SimpleComplex2DArrayObservable;

//=======================================================================

template <class T>
inline NoBinning<T>::NoBinning(uint32_t)
{
  reset();
}


template <class T>
inline void NoBinning<T>::reset(bool forthermalization)
{
  AbstractBinning<T>::reset(forthermalization);
  thermal_count_= (forthermalization ? count_ : 0);
    
  sum_=0;
  sum2_=0;
  count_=0;
  
  min_ = obs_value_traits<T>::max();
  max_ = -obs_value_traits<T>::max();
}


template <class T>
typename NoBinning<T>::convergence_type NoBinning<T>::converged_errors() const
{
  convergence_type conv;
  obs_value_traits<T>::resize_same_as(conv,sum_);
  typename obs_value_traits<convergence_type>::slice_iterator it;
  
  for (it= obs_value_traits<convergence_type>::slice_begin(conv); 
       it!= obs_value_traits<convergence_type>::slice_end(conv); ++it)
    obs_value_traits<convergence_type>::slice_value(conv,it) = CONVERGED;
  return conv;
}

template <class T>
void NoBinning<T>::operator<<(const T& x) 
{
  if(count_==0 && thermal_count_==0)
  {
    obs_value_traits<T>::resize_same_as(sum_,x);
    obs_value_traits<T>::resize_same_as(sum2_,x);
    obs_value_traits<T>::resize_same_as(max_,x);
    obs_value_traits<T>::resize_same_as(min_,x);
  }
  
  if(obs_value_traits<T>::size(x)!=obs_value_traits<T>::size(sum_))
    boost::throw_exception(std::runtime_error("Size of argument does not match in SimpleBinning<T>::add"));

  value_type y=x*x;
  sum_+=x;
  sum2_+=y;

  obs_value_traits<T>::check_for_max(max_,x);
  obs_value_traits<T>::check_for_min(min_,x);

  count_++;
}

template <class T>  
inline typename NoBinning<T>::result_type NoBinning<T>::mean() const 
{
  typedef typename obs_value_traits<T>::count_type count_type;

  if (count())
    return obs_value_cast<result_type,value_type>(sum_)/count_type(count());
  else
    boost::throw_exception(NoMeasurementsError());
  return result_type();
}

template <class T>
inline typename NoBinning<T>::result_type NoBinning<T>::variance() const
{
  typedef typename obs_value_traits<T>::count_type count_type;

  if(count()==0)
    boost::throw_exception(NoMeasurementsError());
    
  if(count_<2) 
    {
      result_type retval;
      obs_value_traits<T>::resize_same_as(retval,sum_);
      retval=inf();
      return retval;
    } // no data collected
  result_type tmp(obs_value_cast<result_type,value_type>(sum_));
  tmp *=tmp/ count_type(count_);
  tmp=obs_value_cast<result_type,value_type>(sum2_) - tmp;
  return tmp/ count_type(count_-1);
}

template <class T>
inline typename NoBinning<T>::result_type NoBinning<T>::error() const
{
  using std::sqrt;
  using alps::sqrt;
  result_type tmp(variance());
  tmp /= count_type(count());
  
  return sqrt(tmp);
}

template <class T>
inline void NoBinning<T>::output_scalar(std::ostream& out) const
{
  if(count()) 
      out << ": " << mean() << " +/- " << error() << std::endl;
}

template <class T> template <class L>
inline void NoBinning<T>::output_vector(std::ostream& out, const L& label) const
{
  if(count()) {
    result_type mean_(mean());
    result_type error_(error());
        
    out << ":\n";
    typename obs_value_traits<L>::slice_iterator it2=obs_value_traits<L>::slice_begin(label);
    for (typename obs_value_traits<result_type>::slice_iterator sit=
           obs_value_traits<result_type>::slice_begin(mean_);
          sit!=obs_value_traits<result_type>::slice_end(mean_);++sit,++it2) {
      std::string lab=obs_value_traits<L>::slice_value(label,it2);
      if (lab=="")
        lab=obs_value_traits<result_type>::slice_name(mean_,sit);
      out << "Entry[" << lab << "]: "
          << obs_value_traits<result_type>::slice_value(mean_,sit) << " +/- " 
          << obs_value_traits<result_type>::slice_value(error_,sit) << std::endl;
    }
  }
}

#ifndef ALPS_WITHOUT_OSIRIS

template <class T>
inline void NoBinning<T>::save(ODump& dump) const
{
  AbstractBinning<T>::save(dump);
  dump << sum_ << sum2_ << count_ << thermal_count_ << min_ << max_;
}

template <class T>
inline void NoBinning<T>::load(IDump& dump) 
{
  AbstractBinning<T>::load(dump);
  dump >> sum_ >> sum2_ >> count_ >> thermal_count_ >> min_ >> max_;
}

#endif

} // end namespace alps

#endif // ALPS_ALEA_NOBINNING_H
