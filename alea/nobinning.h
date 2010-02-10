/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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
#include <alps/math.hpp>
#include <boost/config.hpp>

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

  void reset(bool=true);

  inline void operator<<(const value_type& x);

  result_type mean() const;
  result_type variance() const;
  result_type error() const;
  convergence_type converged_errors() const;

  uint32_t count() const { return  count_;}

  void output_scalar(std::ostream& out) const;
  template <class L> void output_vector(std::ostream& out, const L& l) const;
#ifdef ALPS_HAVE_HDF5
  template<typename E> void read_hdf5 (const E &engine);
  template<typename E> void write_hdf5 (const E &engine)const;
#endif
#ifndef ALPS_WITHOUT_OSIRIS
  void save(ODump& dump) const;
  void load(IDump& dump);
#endif

#ifdef ALPS_HAVE_HDF5
    void serialize(hdf5::iarchive &, bool = false);
    void serialize(hdf5::oarchive &, bool = false) const;
#endif

 private:
    value_type sum_;       // sum of measurements
    value_type sum2_;      // sum of squared measurements
    uint32_t count_;          // total number of measurements
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template <class T> const bool NoBinning<T>::has_tau;
#endif

typedef SimpleObservable<int32_t,NoBinning<int32_t> > SimpleIntObservable;
typedef SimpleObservable<double,NoBinning<double> > SimpleRealObservable;
typedef SimpleObservable<float,NoBinning<float> > SimpleFloatObservable;
typedef SimpleObservable<std::complex<double>,NoBinning<std::complex<double> > > SimpleComplexObservable;
#ifdef ALPS_HAVE_VALARRAY
typedef SimpleObservable< std::valarray<int32_t> , NoBinning<std::valarray<int32_t> > > SimpleIntVectorObservable;
typedef SimpleObservable< std::valarray<double> , NoBinning<std::valarray<double> > > SimpleRealVectorObservable;
typedef SimpleObservable< std::valarray<float> , NoBinning<std::valarray<float> > > SimpleFloatVectorObservable;
typedef SimpleObservable< std::valarray<std::complex<double> > ,
                         NoBinning<std::valarray<std::complex<double> > > > SimpleComplexVectorObservable;
#endif

//=======================================================================

template <class T>
inline NoBinning<T>::NoBinning(uint32_t)
 : count_(0)
{
  reset();
}

template <class T>
inline void NoBinning<T>::reset(bool)
{
  sum_=0;
  sum2_=0;
  count_=0;

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
  if(count_==0)
  {
    obs_value_traits<T>::resize_same_as(sum_,x);
    obs_value_traits<T>::resize_same_as(sum2_,x);
  }

  if(obs_value_traits<T>::size(x)!=obs_value_traits<T>::size(sum_))
    boost::throw_exception(std::runtime_error("Size of argument does not match in NoBinning<T>::add"));

  value_type y=x*x;
  sum_+=x;
  sum2_+=y;

  count_++;
}

template <class T>
inline typename NoBinning<T>::result_type NoBinning<T>::mean() const
{
  typedef typename obs_value_traits<T>::count_type count_type;

  if (count())
    return obs_value_traits<result_type>::convert(sum_)/count_type(count());
  else
    boost::throw_exception(NoMeasurementsError());
  return result_type();
}

template <class T>
inline typename NoBinning<T>::result_type NoBinning<T>::variance() const
{
  using std::abs;
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
  result_type tmp(obs_value_traits<result_type>::convert(sum_));
  tmp *= tmp/ count_type(count_);
  tmp = obs_value_traits<result_type>::convert(sum2_) - tmp;
  obs_value_traits<result_type>::fix_negative(tmp);
  return tmp / count_type(count_-1);
}

template <class T>
inline typename NoBinning<T>::result_type NoBinning<T>::error() const
{
  using std::sqrt;
  result_type tmp(variance());
  tmp /= count_type(count());

  return sqrt(tmp);
}

template <class T>
inline void NoBinning<T>::output_scalar(std::ostream& out) const
{
  if(count()) {
    out << ": " << alps::round<2>(mean()) << " +/- " << alps::round<2>(error());
    if (alps::is_nonzero<2>(error()) && error_underflow(mean(),error()))
      out << " Warning: potential error underflow. Errors might be smaller";
    out << std::endl;
  }
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
          << alps::round<2>(obs_value_traits<result_type>::slice_value(mean_,sit)) << " +/- "
          << alps::round<2>(obs_value_traits<result_type>::slice_value(error_,sit));
      if (alps::is_nonzero<2>(obs_value_traits<result_type>::slice_value(error_,sit)) &&
          error_underflow(obs_value_traits<result_type>::slice_value(mean_,sit),
                          obs_value_traits<result_type>::slice_value(error_,sit)))
      out << " Warning: potential error underflow. Errors might be smaller";
      out << std::endl;
    }
  }
}

#ifndef ALPS_WITHOUT_OSIRIS

template <class T>
inline void NoBinning<T>::save(ODump& dump) const
{
  AbstractBinning<T>::save(dump);
  dump << sum_ << sum2_ << count_;
}

template <class T>
inline void NoBinning<T>::load(IDump& dump)
{
  uint32_t thermal_count_; 
  value_type min_,max_; 
  AbstractBinning<T>::load(dump);
  if(dump.version() >= 306 || dump.version() == 0 /* version is not set */) {
    dump >> sum_ >> sum2_ >> count_;
  }
  else if(dump.version() >= 302)
    dump >> sum_ >> sum2_ >> count_ >> thermal_count_ >> min_ >> max_;
  else
    dump >> sum_ >> sum2_ >> count_ >> thermal_count_ >> min_ >> max_;
}

#endif

#ifdef ALPS_HAVE_HDF5
    template <class T> inline void NoBinning<T>::serialize(hdf5::iarchive & ar, bool write_all_clones) {
      ar
          >> make_pvp("count", count_)
      ;
      if (count_ > 0)
          ar
              >> make_pvp("sum", sum_)
              >> make_pvp("sum2", sum2_)
          ;
    }
    template <class T> inline void NoBinning<T>::serialize(hdf5::oarchive & ar, bool write_all_clones) const {
      ar
          << make_pvp("sum", sum_)
          << make_pvp("sum2", sum2_)
          << make_pvp("count", count_)
      ;
    }
#endif

} // end namespace alps

#endif // ALPS_ALEA_NOBINNING_H
