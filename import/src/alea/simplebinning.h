/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2012 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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
#include <alps/alea/abstractbinning.h>
#include <alps/alea/nan.h>
#include <alps/xml.h>
#include <alps/type_traits/change_value_type.hpp>
#include <alps/type_traits/average_type.hpp>
#include <alps/type_traits/slice.hpp>
#include <alps/numeric/checked_divide.hpp>
#include <alps/numeric/set_negative_0.hpp>
#include <alps/utility/numeric_cast.hpp>
#include <alps/utility/resize.hpp>
#include <alps/utility/size.hpp>
#include <alps/numeric/round.hpp>
#include <alps/numeric/is_nonzero.hpp>

#include <boost/config.hpp>

// workaround for FCC
#ifdef __FCC_VERSION
using std::abs;
#endif

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
  typedef typename change_value_type<T,double>::type time_type;
  typedef std::size_t size_type;
  typedef double count_type;
  typedef typename average_type<T>::type result_type;
  typedef typename change_value_type<T,int>::type convergence_type;

  BOOST_STATIC_CONSTANT(bool, has_tau=true);
  BOOST_STATIC_CONSTANT(int, magic_id=2);

  SimpleBinning(std::size_t=0);

  void reset(bool =true);

  void operator<<(const T& x);

  uint64_t count() const {return count_;} // number of measurements performed
  result_type mean() const;
  result_type variance() const;
  result_type error(std::size_t bin_used=std::numeric_limits<std::size_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ()) const;

  //valarray specializations
  double error_element( std::size_t element, std::size_t bin_used=std::numeric_limits<std::size_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ()) const;
  double binmean_element(std::size_t element, std::size_t i) const ;
  double binvariance_element(std::size_t element, std::size_t i) const;
  double variance_element(std::size_t element) const;

  convergence_type converged_errors() const;
  time_type tau() const; // autocorrelation time

  uint32_t binning_depth() const;
    // depth of logarithmic binning hierarchy = log2(measurements())

  std::size_t size() const { return sum_.size()==0 ? 0 : alps::size(sum_[0]);}

  void output_scalar(std::ostream& out) const;
  template <class L> void output_vector(std::ostream& out, const L&) const;
  void save(ODump& dump) const;
  void load(IDump& dump);

  void save(hdf5::archive &) const;
  void load(hdf5::archive &);

  std::string evaluation_method() const { return "binning";}

  void write_scalar_xml(oxstream& oxs) const;
  template <class IT> void write_vector_xml(oxstream& oxs, IT) const;
private:
  std::vector<result_type> sum_; // sum of measurements in the bin
  std::vector<result_type> sum2_; // sum of the squares
  std::vector<uint64_t> bin_entries_; // number of measurements
  std::vector<result_type> last_bin_; // the last value measured

  uint64_t count_; // total number of measurements (=bin_entries_[0])

  // some fast inlined functions without any range checks
  result_type binmean(std::size_t i) const ;
  result_type binvariance(std::size_t i) const;
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template <class T> const bool SimpleBinning<T>::has_tau;
#endif

template <class T>
inline SimpleBinning<T>::SimpleBinning(std::size_t)
 : count_(0)
{
}

// Reset the observable.
template <class T>
inline void SimpleBinning<T>::reset(bool )
{
  sum_.clear();
  sum2_.clear();
  bin_entries_.clear();
  last_bin_.clear();
  count_ = 0;
}

// add a new measurement
template <> void SimpleBinning<std::valarray<double> >::operator<<(const std::valarray<double>& x);
template <class T>
inline void SimpleBinning<T>::operator<<(const T& x)
{
  // set sizes if starting additions
  if(count_==0)
  {
    last_bin_.resize(1);
    sum_.resize(1);
    sum2_.resize(1);
    bin_entries_.resize(1);
    resize_same_as(last_bin_[0],x);
    resize_same_as(sum_[0],x);
    resize_same_as(sum2_[0],x);
  }
  if(alps::size(x)!=size()) {
    std::cerr << "Size is " << size() << " while new size is " << alps::size(x) << "\n";
    boost::throw_exception(std::runtime_error("Size of argument does not match in SimpleBinning<T>::add"));
  }

  // store x, x^2
  last_bin_[0]=alps::numeric_cast<result_type>(x);
  sum_[0]+=alps::numeric_cast<result_type>(x);
  sum2_[0]+=alps::numeric_cast<result_type>(x)*alps::numeric_cast<result_type>(x);

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

            resize_same_as(last_bin_[bin],x);
            resize_same_as(sum_[bin],x);
            resize_same_as(sum2_[bin],x);
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
  // set sizes if starting additions
  if(count_==0)
  {
    last_bin_.resize(1);
    sum_.resize(1);
    sum2_.resize(1);
    bin_entries_.resize(1);
    resize_same_as(last_bin_[0],x);
    resize_same_as(sum_[0],x);
    resize_same_as(sum2_[0],x);
  }

  if(alps::size(x)!=size()) {
    std::cerr << "Size is " << size() << " while new size is " << alps::size(x) << "\n";
    boost::throw_exception(std::runtime_error("Size of argument does not match in SimpleBinning<T>::add"));
  }

  // store x, x^2 and the minimum and maximum value
  for(std::size_t i=0;i<size();++i){
    last_bin_[0][i]=x[i];
    sum_[0][i]+=x[i];
    sum2_[0][i]+=x[i]*x[i];
  }

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

            resize_same_as(last_bin_[bin],x);
            resize_same_as(sum_[bin],x);
            resize_same_as(sum2_[bin],x);
          }

          result_type x1=(sum_[0]-sum_[bin]);
          x1/=count_type(binlen);

          last_bin_[bin]=x1;

          x1 *= x1;
          sum2_[bin] += x1;
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
  resize_same_as(conv,err);
  const unsigned int range=4;
  typename slice_index<convergence_type>::type it;
  if (binning_depth()<range) {
    for (it= slices(conv).first; it!= slices(conv).second; ++it)
      slice_value(conv,it) = MAYBE_CONVERGED;
  }
  else {
    for (it= slices(conv).first; it!= slices(conv).second; ++it)
      slice_value(conv,it) = CONVERGED;

    for (unsigned int i=binning_depth()-range;i<binning_depth()-1;++i) {
      result_type this_err(error(i));
      for (it= slices(conv).first; it!= slices(conv).second; ++it)
        if (std::abs(slice_value(this_err,it)) >= std::abs(slice_value(err,it)))
          slice_value(conv,it)=CONVERGED;
        else if (std::abs(slice_value(this_err,it)) < 0.824 * std::abs(slice_value(err,it)))
          slice_value(conv,it)=NOT_CONVERGED;
        else if (std::abs(slice_value(this_err,it)) <0.9* std::abs(slice_value(err,it))  &&
            slice_value(conv,it)!=NOT_CONVERGED)
          slice_value(conv,it)=MAYBE_CONVERGED;
    }
  }
  return conv;
}


template <class T>
inline typename SimpleBinning<T>::result_type SimpleBinning<T>::binmean(std::size_t i) const
{
  return sum_[i]/(count_type(bin_entries_[i]) * count_type(1ll<<i));
}
template <class T>
inline double SimpleBinning<T>::binmean_element(std::size_t element, std::size_t i) const
{
  boost::throw_exception(std::invalid_argument("binmean_element only defined for std::valarray<double>"));
  abort();
  return 0;
}
template <>
inline double SimpleBinning<std::valarray<double> >::binmean_element(std::size_t element, std::size_t i) const
{
  return sum_[i][element]/(count_type(bin_entries_[i]) * count_type(1ll<<i));
}


template <class T>
inline typename SimpleBinning<T>::result_type SimpleBinning<T>::binvariance(std::size_t i) const
{
  result_type retval(sum2_[i]);
  retval/=count_type(bin_entries_[i]);
  retval-=binmean(i)*binmean(i);
  return retval;
}

template <class T>
inline double SimpleBinning<T>::binvariance_element(std::size_t element, std::size_t i) const
{
  boost::throw_exception(std::invalid_argument("binmean_element only defined for std::valarray<double>"));
  abort();
  return 0;
}
template <>inline double SimpleBinning<std::valarray<double> >::binvariance_element(std::size_t element, std::size_t i) const
{
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
  if (count()==0)
     boost::throw_exception(NoMeasurementsError());

  if(count()<2)
    {
      result_type retval;
      resize_same_as(retval,sum_[0]);
      retval=inf();
      return retval;
    }
  result_type tmp(sum_[0]);
  tmp *= tmp/count_type(count());
  tmp = sum2_[0] -tmp;
  numeric::set_negative_0(tmp);
  return tmp/count_type(count()-1);
}
// VARIANCE for an element of a vector
template <class T>
inline double SimpleBinning<T>::variance_element(std::size_t element) const
{
  boost::throw_exception(std::invalid_argument("variance_element only defined for std::valarray<double>"));
  abort();
  return 0;
}
template <>
inline double SimpleBinning<std::valarray<double> >::variance_element(std::size_t element) const
{
  if (count()==0)
     boost::throw_exception(NoMeasurementsError());

  if(count()<2)
    {
      return std::numeric_limits<double>::infinity();

    }
  double tmp(sum_[0][element]);
  tmp *= tmp/count_type(count());
  tmp = sum2_[0][element] -tmp;
  numeric::set_negative_0(tmp);
  return tmp/count_type(count()-1);
}

// error estimated from bin i, or from default bin if <0
template <class T>
inline typename SimpleBinning<T>::result_type SimpleBinning<T>::error(std::size_t i) const
{
  if (count()==0)
     boost::throw_exception(NoMeasurementsError());

  if (i==std::numeric_limits<std::size_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ())
    i=binning_depth()-1;

  if (i > binning_depth()-1)
   boost::throw_exception(std::invalid_argument("invalid bin  in SimpleBinning<T>::error"));

  uint64_t binsize_ = bin_entries_[i];

  result_type correction = numeric::checked_divide(binvariance(i),binvariance(0));
  using std::sqrt;
  correction *=(variance()/count_type(binsize_-1));
  return sqrt(correction);
}

template <class T>
inline double SimpleBinning<T>::error_element(std::size_t element,std::size_t i) const
{
  boost::throw_exception(std::invalid_argument("error_element only defined for std::valarray<double>"));
  abort();
}
// error estimated from bin i, or from default bin if <0
template <> inline double SimpleBinning<class std::valarray<double> >::error_element( std::size_t element,std::size_t i) const
{
  if (count()==0)
     boost::throw_exception(NoMeasurementsError());

  if (i==std::numeric_limits<std::size_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ())
    i=binning_depth()-1;

  if (i > binning_depth()-1)
   boost::throw_exception(std::invalid_argument("invalid bin  in SimpleBinning<T>::error"));

  uint64_t binsize_ = bin_entries_[i];

  double correction = binvariance_element(element,i)/binvariance_element(element,0);
  correction *=(variance_element(element)/count_type(binsize_-1));
  return std::sqrt(correction);
}

template <class T>
inline typename SimpleBinning<T>::time_type SimpleBinning<T>::tau() const
{
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
    resize_same_as(retval,sum_[0]);
    retval=inf();
    return retval;
  }
}


template <class T>
void SimpleBinning<T>::output_scalar(std::ostream& out) const
{
  if(count())
  {
    out << ": " << std::setprecision(6) << alps::numeric::round<2>(mean()) << " +/- "
        << std::setprecision(3) << alps::numeric::round<2>(error()) << "; tau = "
        << std::setprecision(3) << (alps::numeric::is_nonzero<2>(error()) ? tau() : 0)
        << std::setprecision(6);
    if (alps::numeric::is_nonzero<2>(error())) {
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
            << " : " << std::setw(8) << count()/(1ll<<i)
            << " entries: error = " << alps::numeric::round<2>(error(i)) << std::endl;
      out.setf(oldflags);
    }
  }
}

template <class T>
void SimpleBinning<T>::write_scalar_xml(oxstream& oxs) const {
  for (unsigned int i = 0; i < binning_depth(); ++i) {
    int prec=int(4-std::log10(std::abs(error(i)/binmean(i))));
    prec = (prec>=3 && prec<20 ? prec : 16);
    oxs << start_tag("BINNED") << attribute("size",boost::lexical_cast<std::string>(1ll<<i))
        << no_linebreak << start_tag("COUNT") << count()/(1ll<<i) << end_tag("COUNT")
        << start_tag("MEAN") << attribute("method", "simple") << no_linebreak << precision(binmean(i), prec) << end_tag("MEAN")
        << start_tag("ERROR") << attribute("method", "simple")
        << no_linebreak << precision(error(i), 3) << end_tag("ERROR")
        << end_tag("BINNED");
  }
}

template <class T> template <class IT>
void SimpleBinning<T>::write_vector_xml(oxstream& oxs, IT it) const {
  for (int i = 0; i < (int)binning_depth() ; ++i) {
    int prec=int(4-std::log10(std::abs(slice_value(error(i),it)
                            /slice_value(binmean(i),it))));
    prec = (prec>=3 && prec<20 ? prec : 16);
    oxs << start_tag("BINNED") << attribute("size",boost::lexical_cast<std::string>(1ll<<i))
              << no_linebreak << start_tag("COUNT") << count()/(1ll<<i) << end_tag("COUNT")
        << start_tag("MEAN") << attribute("method", "simple")
        << no_linebreak << precision(slice_value(binmean(i),it), 8) << end_tag("MEAN")
        << start_tag("ERROR") << attribute("method", "simple")
        << no_linebreak << precision(slice_value(error(i),it), 3) << end_tag("ERROR")
        << end_tag("BINNED");
  }
}

template <> template <class IT>
void SimpleBinning<std::valarray<double> >::write_vector_xml(oxstream& oxs, IT it) const {
  for (int i = 0; i < (int)binning_depth() ; ++i) {
    int prec=int(4-std::log10(std::abs(error_element(it,i)
                            /binmean_element(it,i))));
    prec = (prec>=3 && prec<20 ? prec : 16);
    oxs << start_tag("BINNED") << attribute("size",boost::lexical_cast<std::string>(1ll<<i))
              << no_linebreak << start_tag("COUNT") << count()/(1ll<<i) << end_tag("COUNT")
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
    typename alps::slice_index<L>::type it2=slices(label).first;
    for (typename alps::slice_index<result_type>::type sit= slices(mean_).first; 
         sit!=slices(mean_).second;++sit,++it2)
    {
      std::string lab = slice_value(label,it2);
      if (lab=="")
        lab = slice_name(mean_,sit);
      out << "Entry[" << lab << "]: "
          << alps::numeric::round<2>(slice_value(mean_,sit)) << " +/- "
          << alps::numeric::round<2>(slice_value(error_,sit))
          << "; tau = " << (alps::numeric::is_nonzero<2>(slice_value(error_,sit)) ? slice_value(tau_,sit) : 0);
      if (alps::numeric::is_nonzero<2>(slice_value(error_,sit))) {
        if (slice_value(conv_,sit)==MAYBE_CONVERGED)
          out << " WARNING: check error convergence";
        if (slice_value(conv_,sit)==NOT_CONVERGED)
          out << " WARNING: ERRORS NOT CONVERGED!!!";
        if (error_underflow(slice_value(mean_,sit),slice_value(error_,sit)))
          out << " Warning: potential error underflow. Errors might be smaller";
      }
      out << std::endl;
      if (binning_depth()>1)
        {
          // detailed errors
          std::ios::fmtflags oldflags = out.setf(std::ios::left,std::ios::adjustfield);
          for(int i=0;i<(int)binning_depth();i++)
            out << "    bin #" << std::setw(3) <<  i+1
                << " : " << std::setw(8) << count()/(1ll<<i)
                << " entries: error = "
                << slice_value(errs_[i],sit)
                << std::endl;
          out.setf(oldflags);
        }
    }
  }
}

template <class T>
inline void SimpleBinning<T>::save(ODump& dump) const
{
  AbstractBinning<T>::save(dump);
    dump << sum_ << sum2_ << bin_entries_ << last_bin_ << count_;
}

template <class T>
inline void SimpleBinning<T>::load(IDump& dump)
{
  // local variables for depreacted members
  // bool has_minmax_;
  value_type min_, max_;
  uint32_t thermal_count_;

  AbstractBinning<T>::load(dump);
  if(dump.version() >= 306 || dump.version() == 0 /* version is not set */){
    //previously saved min and max will be ignored from now on.
    dump >> sum_ >> sum2_ >> bin_entries_ >> last_bin_ >> count_;
  }
  else if(dump.version() >= 302 /* version is not set */){
    //previously saved min and max will be ignored from now on.
    dump >> sum_ >> sum2_ >> bin_entries_ >> last_bin_ >> count_ >> thermal_count_
         >> min_>> max_;
  }
  else {
    // some data types have changed from 32 to 64 Bit between version 301 and 302
    uint32_t count_tmp;
    std::vector<uint32_t> bin_entries_tmp;
    dump >> sum_ >> sum2_ >> bin_entries_tmp >> last_bin_ >> count_tmp >> thermal_count_
         >> min_ >> max_;
    // perform the conversions which may be necessary
    count_ = count_tmp;
    bin_entries_.assign(bin_entries_tmp.begin(), bin_entries_tmp.end());
   }
}

template <class T> inline void SimpleBinning<T>::save(hdf5::archive & ar) const {
    ar
        << make_pvp("count", count_)
        << make_pvp("timeseries/logbinning", sum_)
        << make_pvp("timeseries/logbinning/@binningtype", "logarithmic")
        << make_pvp("timeseries/logbinning2", sum2_)
        << make_pvp("timeseries/logbinning2/@binningtype", "logarithmic")
        << make_pvp("timeseries/logbinning_lastbin", last_bin_)
        << make_pvp("timeseries/logbinning_lastbin/@binningtype", "logarithmic")
        << make_pvp("timeseries/logbinning_counts", bin_entries_)
        << make_pvp("timeseries/logbinning_counts/@binningtype", "logarithmic")
    ;
    if (sum_.size() && sum2_.size())
        ar
            << make_pvp("sum", sum_[0])
            << make_pvp("sum2", sum2_[0])
        ;
}
template <class T> inline void SimpleBinning<T>::load(hdf5::archive & ar) {
    ar
        >> make_pvp("count", count_)
        >> make_pvp("timeseries/logbinning", sum_)
        >> make_pvp("timeseries/logbinning2", sum2_)
        >> make_pvp("timeseries/logbinning_lastbin", last_bin_)
        >> make_pvp("timeseries/logbinning_counts", bin_entries_)
    ;
}

} // end namespace alps

#endif // ALPS_ALEA_SIMPLEBINNING_H
