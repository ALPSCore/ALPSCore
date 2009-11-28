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

#ifndef ALPS_ALEA_SIMPLEBINNING_H
#define ALPS_ALEA_SIMPLEBINNING_H

#include <alps/config.h>
#include <alps/alea/obsvalue.h>
#include <alps/alea/abstractbinning.h>
#include <alps/alea/nan.h>
#include <alps/math.hpp>
#include <alps/xml.h>
#include <boost/config.hpp>

//=======================================================================
// SimpleBinning
//
// simple binning strategy
//-----------------------------------------------------------------------

namespace alps {

template <class T=double>
class SimpleBinning : public AbstractBinning<T>
{
  typedef AbstractBinning<T> super_type;
 public:
  typedef T value_type;
  typedef typename obs_value_traits<T>::time_type time_type;
  typedef typename obs_value_traits<T>::size_type size_type;
  typedef typename obs_value_traits<T>::count_type count_type;
  typedef typename obs_value_traits<T>::result_type result_type;
  typedef typename obs_value_traits<T>::convergence_type convergence_type;

  BOOST_STATIC_CONSTANT(bool, has_tau=true);
  BOOST_STATIC_CONSTANT(int, magic_id=2);

  SimpleBinning(uint32_t=0);
  virtual ~SimpleBinning() {}

  void reset(bool forthermalization=false);
  void operator<<(const T& x);

  uint64_t count() const {return super_type::is_thermalized() ? count_ : 0;} // number of measurements performed
  result_type mean() const;
  result_type variance() const;
  result_type error(uint64_t bin_used=std::numeric_limits<uint64_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ()) const;

  //valarray specializations
  double error_element( uint64_t element, uint64_t bin_used=std::numeric_limits<uint64_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ()) const;
  double binmean_element(uint64_t element, uint64_t i) const ;
  double binvariance_element(uint64_t element, uint64_t i) const;
  double variance_element(uint64_t element) const;

  convergence_type converged_errors() const;
  time_type tau() const; // autocorrelation time

  uint32_t binning_depth() const;
    // depth of logarithmic binning hierarchy = log2(measurements())

  //min and max are expensive to compute. This interface is now disabled.
  value_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const {return min_;}
  value_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const {return max_;}

  uint64_t get_thermalization() const { return super_type::is_thermalized() ? thermal_count_ : count_;}

  uint64_t size() const { return sum_.size()==0?0:obs_value_traits<T>::size(sum_[0]);}

  void output_scalar(std::ostream& out) const;
  template <class L> void output_vector(std::ostream& out, const L&) const;
#ifndef ALPS_WITHOUT_OSIRIS
  virtual void save(ODump& dump) const;
  virtual void load(IDump& dump);
#endif

  std::string evaluation_method() const { return "binning";}

  void write_scalar_xml(oxstream& oxs) const;
  template <class IT> void write_vector_xml(oxstream& oxs, IT) const;
#ifdef ALPS_HAVE_HDF5
  template<typename E> void read_hdf5 (const E &engine);
  template<typename E> void write_hdf5 (E &engine) const;
#endif
private:
  std::vector<result_type> sum_; // sum of measurements in the bin
  std::vector<result_type> sum2_; // sum of the squares
  std::vector<uint64_t> bin_entries_; // number of measurements
  std::vector<result_type> last_bin_; // the last value measured

  uint64_t count_; // total number of measurements (=bin_entries_[0])
  uint32_t thermal_count_; // meaurements performed during thermalization
  value_type min_,max_; // minimum and maximum value

  // some fast inlined functions without any range checks
  result_type binmean(uint64_t i) const ;
  result_type binvariance(uint64_t i) const;
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template <class T> const bool SimpleBinning<T>::has_tau;
#endif

template <class T>
inline SimpleBinning<T>::SimpleBinning(uint32_t)
{
  reset();
}


// Reset the observable.
template <class T>
inline void SimpleBinning<T>::reset(bool forthermalization)
{
  typedef typename obs_value_traits<T>::count_type count_type;

  AbstractBinning<T>::reset(forthermalization);

  thermal_count_= (forthermalization ? count_ : 0);

  sum_.clear();
  sum2_.clear();
  bin_entries_.clear();
  last_bin_.clear();

  count_ = 0;

  min_ =  obs_value_traits<T>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
  max_ = -obs_value_traits<T>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
}


// add a new measurement
template <> void SimpleBinning<std::valarray<double> >::operator<<(const std::valarray<double>& x);
template <class T>
inline void SimpleBinning<T>::operator<<(const T& x)
{
  typedef typename obs_value_traits<T>::count_type count_type;

  // set sizes if starting additions
  if(count_==0)
  {
    last_bin_.resize(1);
    sum_.resize(1);
    sum2_.resize(1);
    bin_entries_.resize(1);
    obs_value_traits<result_type>::resize_same_as(last_bin_[0],x);
    obs_value_traits<result_type>::resize_same_as(sum_[0],x);
    obs_value_traits<result_type>::resize_same_as(sum2_[0],x);
    obs_value_traits<result_type>::resize_same_as(max_,x);
    obs_value_traits<result_type>::resize_same_as(min_,x);
  }

  if(obs_value_traits<T>::size(x)!=size()) {
    std::cerr << "Size is " << size() << " while new size is " << obs_value_traits<T>::size(x) << "\n";
    boost::throw_exception(std::runtime_error("Size of argument does not match in SimpleBinning<T>::add"));
  }

  // store x, x^2 and the minimum and maximum value
  last_bin_[0]=obs_value_traits<result_type>::convert(x);
  sum_[0]+=obs_value_traits<result_type>::convert(x);
  sum2_[0]+=obs_value_traits<result_type>::convert(x)*obs_value_traits<result_type>::convert(x);
  //obs_value_traits<T>::check_for_max(max_,x);
  //obs_value_traits<T>::check_for_min(min_,x);

  uint64_t i=count_;
  count_++;
  bin_entries_[0]++;
  uint64_t binlen=1;
  std::size_t bin=0;

  // binning
  do
    {
      if(i&1)
        {
          // a bin is filled
          binlen*=2;
          bin++;
          if(bin>=last_bin_.size())
          {
            last_bin_.resize(std::max BOOST_PREVENT_MACRO_SUBSTITUTION (bin+1,last_bin_.size()));
            sum_.resize(std::max BOOST_PREVENT_MACRO_SUBSTITUTION (bin+1, sum_.size()));
            sum2_.resize(std::max BOOST_PREVENT_MACRO_SUBSTITUTION (bin+1,sum2_.size()));
            bin_entries_.resize(std::max BOOST_PREVENT_MACRO_SUBSTITUTION (bin+1,bin_entries_.size()));

            obs_value_traits<result_type>::resize_same_as(last_bin_[bin],x);
            obs_value_traits<result_type>::resize_same_as(sum_[bin],x);
            obs_value_traits<result_type>::resize_same_as(sum2_[bin],x);
          }

          result_type x1=(sum_[0]-sum_[bin]);
          x1/=count_type(binlen);

          result_type y1 = x1*x1;

          last_bin_[bin]=x1;
          sum2_[bin] += y1;
          sum_[bin] = sum_[0];
          bin_entries_[bin]++;
        }
      else
        break;
    } while ( i>>=1);
}

template <> inline void SimpleBinning<std::valarray<double> >::operator<<(const std::valarray<double> & x)
{
  typedef obs_value_traits<std::valarray<double> >::count_type count_type;

  // set sizes if starting additions
  if(count_==0)
  {
    last_bin_.resize(1);
    sum_.resize(1);
    sum2_.resize(1);
    bin_entries_.resize(1);
    obs_value_traits<result_type>::resize_same_as(last_bin_[0],x);
    obs_value_traits<result_type>::resize_same_as(sum_[0],x);
    obs_value_traits<result_type>::resize_same_as(sum2_[0],x);
    obs_value_traits<result_type>::resize_same_as(max_,x);
    obs_value_traits<result_type>::resize_same_as(min_,x);
  }

  if(obs_value_traits<std::valarray<double> >::size(x)!=size()) {
    std::cerr << "Size is " << size() << " while new size is " << obs_value_traits<std::valarray<double> >::size(x) << "\n";
    boost::throw_exception(std::runtime_error("Size of argument does not match in SimpleBinning<T>::add"));
  }

  // store x, x^2 and the minimum and maximum value
  for(std::size_t i=0;i<size();++i){
    last_bin_[0][i]=x[i];
    sum_[0][i]+=x[i];
    sum2_[0][i]+=x[i]*x[i];
  }
  //obs_value_traits<std::valarray<double> >::check_for_max(max_,x);
  //obs_value_traits<std::valarray<double> >::check_for_min(min_,x);

  uint64_t i=count_;
  count_++;
  bin_entries_[0]++;
  uint64_t binlen=1;
  std::size_t bin=0;

  // binning
  do
    {
      if(i&1)
        {
          // a bin is filled
          binlen*=2;
          bin++;
          if(bin>=last_bin_.size())
          {
            last_bin_.resize(std::max BOOST_PREVENT_MACRO_SUBSTITUTION (bin+1,last_bin_.size()));
            sum_.resize(std::max BOOST_PREVENT_MACRO_SUBSTITUTION (bin+1, sum_.size()));
            sum2_.resize(std::max BOOST_PREVENT_MACRO_SUBSTITUTION (bin+1,sum2_.size()));
            bin_entries_.resize(std::max BOOST_PREVENT_MACRO_SUBSTITUTION (bin+1,bin_entries_.size()));

            obs_value_traits<result_type>::resize_same_as(last_bin_[bin],x);
            obs_value_traits<result_type>::resize_same_as(sum_[bin],x);
            obs_value_traits<result_type>::resize_same_as(sum2_[bin],x);
          }

          result_type x1=(sum_[0]-sum_[bin]);
          x1/=count_type(binlen);

          result_type y1 = x1*x1;

          last_bin_[bin]=x1;
          sum2_[bin] += y1;
          sum_[bin] = sum_[0];
          bin_entries_[bin]++;
        }
      else
        break;
    } while ( i>>=1);
}

template <class T>
inline uint32_t SimpleBinning<T>::binning_depth() const
{
  return ( int(sum_.size())-7 < 1 ) ? 1 : int(sum_.size())-7;
}


template <class T>
typename SimpleBinning<T>::convergence_type SimpleBinning<T>::converged_errors() const
{
  convergence_type conv;
  result_type err=error();
  obs_value_traits<T>::resize_same_as(conv,err);
  const unsigned int range=4;
  typename obs_value_traits<convergence_type>::slice_iterator it;
  if (binning_depth()<range) {
    for (it= obs_value_traits<convergence_type>::slice_begin(conv);
       it!= obs_value_traits<convergence_type>::slice_end(conv); ++it)
      obs_value_traits<convergence_type>::slice_value(conv,it) = MAYBE_CONVERGED;
  }
  else {
    for (it= obs_value_traits<convergence_type>::slice_begin(conv);
       it!= obs_value_traits<convergence_type>::slice_end(conv); ++it)
      obs_value_traits<convergence_type>::slice_value(conv,it) = CONVERGED;

    for (unsigned int i=binning_depth()-range;i<binning_depth()-1;++i) {
      result_type this_err(error(i));
      for (it= obs_value_traits<convergence_type>::slice_begin(conv);
           it!= obs_value_traits<convergence_type>::slice_end(conv); ++it) {
        if (std::abs(obs_value_traits<result_type>::slice_value(this_err,it)) >=
            std::abs(obs_value_traits<result_type>::slice_value(err,it)))
          obs_value_traits<convergence_type>::slice_value(conv,it)=CONVERGED;
        else if (std::abs(obs_value_traits<result_type>::slice_value(this_err,it)) <0.824*
            std::abs(obs_value_traits<result_type>::slice_value(err,it)))
          obs_value_traits<convergence_type>::slice_value(conv,it)=NOT_CONVERGED;
        else if (std::abs(obs_value_traits<result_type>::slice_value(this_err,it)) <0.9*
            std::abs(obs_value_traits<result_type>::slice_value(err,it))  &&
            obs_value_traits<convergence_type>::slice_value(conv,it)!=NOT_CONVERGED)
          obs_value_traits<convergence_type>::slice_value(conv,it)=MAYBE_CONVERGED;
      }
    }
  }
  return conv;
}


template <class T>
inline typename SimpleBinning<T>::result_type SimpleBinning<T>::binmean(uint64_t i) const
{
  typedef typename obs_value_traits<T>::count_type count_type;

  return sum_[i]/(count_type(bin_entries_[i]) * count_type(1<<i));
}
template <class T>
inline double SimpleBinning<T>::binmean_element(uint64_t element, uint64_t i) const
{
  boost::throw_exception(std::invalid_argument("binmean_element only defined for std::valarray<double>"));
  abort();
  return 0;
}
template <>
inline double SimpleBinning<std::valarray<double> >::binmean_element(uint64_t element, uint64_t i) const
{
  typedef obs_value_traits<std::valarray<double> >::count_type count_type;

  return sum_[i][element]/(count_type(bin_entries_[i]) * count_type(1<<i));
}


template <class T>
inline typename SimpleBinning<T>::result_type SimpleBinning<T>::binvariance(uint64_t i) const
{
  typedef typename obs_value_traits<T>::count_type count_type;

  result_type retval(sum2_[i]);
  retval/=count_type(bin_entries_[i]);
  retval-=binmean(i)*binmean(i);
  return retval;
}

template <class T>
inline double SimpleBinning<T>::binvariance_element(uint64_t element, uint64_t i) const
{
  boost::throw_exception(std::invalid_argument("binmean_element only defined for std::valarray<double>"));
  abort();
  return 0;
}
template <>inline double SimpleBinning<std::valarray<double> >::binvariance_element(uint64_t element, uint64_t i) const
{
  typedef  obs_value_traits<std::valarray<double> >::count_type count_type;

  double retval(sum2_[i][element]);
  retval/=count_type(bin_entries_[i]);
  retval-=binmean_element(element,i)*binmean_element(element,i);
  return retval;
}



//---------------------------------------------------------------------
// EVALUATION FUNCTIONS
//---------------------------------------------------------------------

// MEAN
template <class T>
inline typename SimpleBinning<T>::result_type SimpleBinning<T>::mean() const
{
  if (count()==0)
     boost::throw_exception(NoMeasurementsError());
  return sum_[0]/count_type(count());
}


// VARIANCE
template <class T>
inline typename SimpleBinning<T>::result_type SimpleBinning<T>::variance() const
{
  typedef typename obs_value_traits<T>::count_type count_type;

  if (count()==0)
     boost::throw_exception(NoMeasurementsError());

  if(count()<2)
    {
      result_type retval;
      obs_value_traits<T>::resize_same_as(retval,min_);
      retval=inf();
      return retval;
    }
  result_type tmp(sum_[0]);
  tmp *= tmp/count_type(count());
  tmp = sum2_[0] -tmp;
  obs_value_traits<result_type>::fix_negative(tmp);
  return tmp/count_type(count()-1);
}
// VARIANCE for an element of a vector
template <class T>
inline double SimpleBinning<T>::variance_element(uint64_t element) const
{
  boost::throw_exception(std::invalid_argument("variance_element only defined for std::valarray<double>"));
  abort();
  return 0;
}
template <>
inline double SimpleBinning<std::valarray<double> >::variance_element(uint64_t element) const
{
  typedef obs_value_traits<std::valarray<double> >::count_type count_type;

  if (count()==0)
     boost::throw_exception(NoMeasurementsError());

  if(count()<2)
    {
      return std::numeric_limits<double>::infinity();

    }
  double tmp(sum_[0][element]);
  tmp *= tmp/count_type(count());
  tmp = sum2_[0][element] -tmp;
  obs_value_traits<double>::fix_negative(tmp);
  return tmp/count_type(count()-1);
}

// error estimated from bin i, or from default bin if <0
template <class T>
inline typename SimpleBinning<T>::result_type SimpleBinning<T>::error(uint64_t i) const
{
  typedef typename obs_value_traits<T>::count_type count_type;
  if (count()==0)
     boost::throw_exception(NoMeasurementsError());

  if (i==std::numeric_limits<uint64_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ())
    i=binning_depth()-1;

  if (i > binning_depth()-1)
   boost::throw_exception(std::invalid_argument("invalid bin  in SimpleBinning<T>::error"));

  uint64_t binsize_ = bin_entries_[i];

  result_type correction = obs_value_traits<result_type>::check_divide(binvariance(i),binvariance(0));
  using std::sqrt;
  correction *=(variance()/count_type(binsize_-1));
  return sqrt(correction);
}

template <class T>
inline double SimpleBinning<T>::error_element(uint64_t element,uint64_t i) const
{
  boost::throw_exception(std::invalid_argument("error_element only defined for std::valarray<double>"));
  abort();
}
// error estimated from bin i, or from default bin if <0
template <> inline double SimpleBinning<class std::valarray<double> >::error_element( uint64_t element,uint64_t i) const
{
  typedef obs_value_traits<class std::valarray<double> >::count_type count_type;
  if (count()==0)
     boost::throw_exception(NoMeasurementsError());

  if (i==std::numeric_limits<uint64_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ())
    i=binning_depth()-1;

  if (i > binning_depth()-1)
   boost::throw_exception(std::invalid_argument("invalid bin  in SimpleBinning<T>::error"));

  uint64_t binsize_ = bin_entries_[i];

  double correction = binvariance_element(element,i)/binvariance_element(element,0);
  correction *=(variance_element(element)/count_type(binsize_-1));
  return std::sqrt(correction);
}

template <class T>
inline typename obs_value_traits<T>::time_type SimpleBinning<T>::tau() const
{
  typedef typename obs_value_traits<T>::count_type count_type;

  if (count()==0)
     boost::throw_exception(NoMeasurementsError());

  if( binning_depth() >= 2 )
  {
    count_type factor =count()-1;
    time_type er(std::abs(error()));
    er *=er*factor;
    er /= std::abs(variance());
    er -=1.;
    return 0.5*er;
  }
  else
  {
    time_type retval;
    obs_value_traits<T>::resize_same_as(retval,min_);
    retval=inf();
    return retval;
  }
}


template <class T>
void SimpleBinning<T>::output_scalar(std::ostream& out) const
{
  if(count())
  {
    out << ": " << std::setprecision(6) << alps::round<2>(mean()) << " +/- "
        << std::setprecision(3) << alps::round<2>(error()) << "; tau = "
        << std::setprecision(3) << (alps::is_nonzero<2>(error()) ? tau() : 0)
        << std::setprecision(6);
    if (alps::is_nonzero<2>(error())) {
      if (converged_errors()==MAYBE_CONVERGED)
        out << " WARNING: check error convergence";
      if (converged_errors()==NOT_CONVERGED)
        out << " WARNING: ERRORS NOT CONVERGED!!!";
      if (error_underflow(mean(),error()))
        out << " Warning: potential error underflow. Errors might be smaller";
    }
    out << std::endl;
    if (binning_depth()>1)
    {
      // detailed errors
      std::ios::fmtflags oldflags = out.setf(std::ios::left,std::ios::adjustfield);
      for(unsigned int i=0;i<binning_depth();i++)
        out << "    bin #" << std::setw(3) <<  i+1
            << " : " << std::setw(8) << count()/(1<<i)
            << " entries: error = " << alps::round<2>(error(i)) << std::endl;
      out.setf(oldflags);
    }
  }
}

template <class T>
void SimpleBinning<T>::write_scalar_xml(oxstream& oxs) const {
  for (unsigned int i = 0; i < binning_depth(); ++i) {
    int prec=int(4-std::log10(std::abs(error(i)/binmean(i))));
    prec = (prec>=3 && prec<20 ? prec : 16);
    oxs << start_tag("BINNED") << attribute("size",boost::lexical_cast<std::string,int>(1<<i))
        << no_linebreak << start_tag("COUNT") << count()/(1<<i) << end_tag("COUNT")
        << start_tag("MEAN") << attribute("method", "simple") << no_linebreak << precision(binmean(i), prec) << end_tag("MEAN")
        << start_tag("ERROR") << attribute("method", "simple")
        << no_linebreak << precision(error(i), 3) << end_tag("ERROR")
        << end_tag("BINNED");
  }
}

template <class T> template <class IT>
void SimpleBinning<T>::write_vector_xml(oxstream& oxs, IT it) const {
  for (int i = 0; i < (int)binning_depth() ; ++i) {
    int prec=int(4-std::log10(std::abs(obs_value_traits<result_type>::slice_value(error(i),it)
                            /obs_value_traits<result_type>::slice_value(binmean(i),it))));
    prec = (prec>=3 && prec<20 ? prec : 16);
    oxs << start_tag("BINNED") << attribute("size",boost::lexical_cast<std::string,int>(1<<i))
              << no_linebreak << start_tag("COUNT") << count()/(1<<i) << end_tag("COUNT")
        << start_tag("MEAN") << attribute("method", "simple")
        << no_linebreak << precision(obs_value_traits<result_type>::slice_value(binmean(i),it), 8) << end_tag("MEAN")
        << start_tag("ERROR") << attribute("method", "simple")
        << no_linebreak << precision(obs_value_traits<result_type>::slice_value(error(i),it), 3) << end_tag("ERROR")
        << end_tag("BINNED");
  }
}

template <> template <class IT>
void SimpleBinning<std::valarray<double> >::write_vector_xml(oxstream& oxs, IT it) const {
  for (int i = 0; i < (int)binning_depth() ; ++i) {
    int prec=int(4-std::log10(std::abs(error_element(it,i)
                            /binmean_element(it,i))));
    prec = (prec>=3 && prec<20 ? prec : 16);
    oxs << start_tag("BINNED") << attribute("size",boost::lexical_cast<std::string,int>(1<<i))
              << no_linebreak << start_tag("COUNT") << count()/(1<<i) << end_tag("COUNT")
        << start_tag("MEAN") << attribute("method", "simple")
        << no_linebreak << precision(binmean_element(it,i), 8) << end_tag("MEAN")
        << start_tag("ERROR") << attribute("method", "simple")
        << no_linebreak << precision(error_element(it,i), 3) << end_tag("ERROR")
        << end_tag("BINNED");
  }
}


template <class T> template <class L>
inline void SimpleBinning<T>::output_vector(std::ostream& out, const L& label) const
{
  if(count())
  {
    result_type mean_(mean());
    result_type error_(error());
    time_type tau_(tau());
    convergence_type conv_(converged_errors());
    std::vector<result_type> errs_(binning_depth(),error_);
    for (int i=0;i<(int)binning_depth();++i)
      errs_[i]=error(i);

    out << "\n";
    typename obs_value_traits<L>::slice_iterator it2=obs_value_traits<L>::slice_begin(label);
    for (typename obs_value_traits<result_type>::slice_iterator sit=
           obs_value_traits<result_type>::slice_begin(mean_);
          sit!=obs_value_traits<result_type>::slice_end(mean_);++sit,++it2)
    {
      std::string lab=obs_value_traits<L>::slice_value(label,it2);
      if (lab=="")
        lab=obs_value_traits<result_type>::slice_name(mean_,sit);
      out << "Entry[" << lab << "]: "
          << alps::round<2>(obs_value_traits<result_type>::slice_value(mean_,sit)) << " +/- "
          << alps::round<2>(obs_value_traits<result_type>::slice_value(error_,sit))
          << "; tau = " << (alps::is_nonzero<2>(obs_value_traits<result_type>::slice_value(error_,sit)) ? obs_value_traits<time_type>::slice_value(tau_,sit) : 0);
      if (alps::is_nonzero<2>(obs_value_traits<result_type>::slice_value(error_,sit))) {
        if (obs_value_traits<convergence_type>::slice_value(conv_,sit)==MAYBE_CONVERGED)
          out << " WARNING: check error convergence";
        if (obs_value_traits<convergence_type>::slice_value(conv_,sit)==NOT_CONVERGED)
          out << " WARNING: ERRORS NOT CONVERGED!!!";
        if (error_underflow(obs_value_traits<result_type>::slice_value(mean_,sit),
                            obs_value_traits<result_type>::slice_value(error_,sit)))
          out << " Warning: potential error underflow. Errors might be smaller";
      }
      out << std::endl;
      if (binning_depth()>1)
        {
          // detailed errors
          std::ios::fmtflags oldflags = out.setf(std::ios::left,std::ios::adjustfield);
          for(int i=0;i<(int)binning_depth();i++)
            out << "    bin #" << std::setw(3) <<  i+1
                << " : " << std::setw(8) << count()/(1<<i)
                << " entries: error = "
                << obs_value_traits<result_type>::slice_value(errs_[i],sit)
                << std::endl;
          out.setf(oldflags);
        }
    }
  }
}

#ifndef ALPS_WITHOUT_OSIRIS

template <class T>
inline void SimpleBinning<T>::save(ODump& dump) const
{
  AbstractBinning<T>::save(dump);
    dump << sum_ << sum2_ << bin_entries_ << last_bin_ << count_ << thermal_count_<<min_<<max_;
}

template <class T>
inline void SimpleBinning<T>::load(IDump& dump)
{
  AbstractBinning<T>::load(dump);
  if(dump.version() >= 302 || dump.version() == 0 /* version is not set */){
    //previously saved min and max will be ignored from now on.
    dump >> sum_ >> sum2_ >> bin_entries_ >> last_bin_ >> count_ >> thermal_count_
         >> min_>> max_;
  }
  else {
    // some data types have changed from 32 to 64 Bit between version 301 and 302
    uint32_t count_tmp;
    std::vector<uint32_t> bin_entries_tmp;
    //previously saved min and max will be ignored from now on.
    value_type min_ignored;
    value_type max_ignored;
    dump >> sum_ >> sum2_ >> bin_entries_tmp >> last_bin_ >> count_tmp >> thermal_count_
         >> min_ignored >> max_ignored;
    // perform the conversions which may be necessary
    count_ = count_tmp;
    bin_entries_.assign(bin_entries_tmp.begin(), bin_entries_tmp.end());
   }
}
#endif

} // end namespace alps

#ifndef ALPS_WITHOUT_OSIRIS

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template<class T>
inline alps::ODump& operator<<(alps::ODump& od, const alps::SimpleBinning<T>& m)
{ m.save(od); return od; }

template<class T>
inline alps::IDump& operator>>(alps::IDump& id, alps::SimpleBinning<T>& m)
{ m.load(id); return id; }

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif

#endif // ALPS_ALEA_SIMPLEBINNING_H
