/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* Copyright (C) 2011-2012 by Lukas Gamper <gamperl@gmail.com>,
*                            Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Maximilian Poprawe <poprawem@ethz.ch>
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

#ifndef ALPS_ALEA_MCANALYZE_HPP
#define ALPS_ALEA_MCANALYZE_HPP

#include <alps/config.h>
#include <alps/utility/resize.hpp>
#include <alps/utility/set_zero.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/numeric/functional.hpp>
#include <alps/numeric/regression.hpp>
#include <alps/numeric/vector_valarray_conversion.hpp>
#include <alps/numeric/sequence_comparisons.hpp>
#include <alps/type_traits/average_type.hpp>
#include <alps/alea/simplebinning.h>
#include <alps/alea/value_with_error.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/accumulators/numeric/functional/vector.hpp>
#include <boost/parameter.hpp>

#ifdef ALPS_HAVE_PYTHON
  #include <boost/python.hpp>
  #include <alps/python/numpy_array.hpp>
#endif


#include <vector>
#include <iostream>
#include <iterator>
#include <numeric>
#include <limits>
#include <math.h>
#include <exception>




namespace alps {
namespace alea {


// Boost::parameter
BOOST_PARAMETER_NAME(distance)
BOOST_PARAMETER_NAME(limit)

BOOST_PARAMETER_NAME(min)
BOOST_PARAMETER_NAME(max)
BOOST_PARAMETER_NAME(from)
BOOST_PARAMETER_NAME(to)

// object that can be compared to any numeric value and always returns false exept when compared to itself
// needed for boost::parameter argument passing.
class None_class {
public:
  None_class (): is_none(true) {}
  None_class (const None_class& other): is_none(other.is_none) {} 
  template <class T> None_class (const T&): is_none(false) {}

  bool operator== (const None_class& other) const {return (is_none == other.is_none) ;}
  bool operator!= (const None_class& other) const {return !(*this == other);}

  operator int () {return 0;}

private:
  bool is_none;
} None;


// Declarations
template <class ValueType>
class mcdata;

template <class ValueType>
class mctimeseries;

template <class ValueType>
class mctimeseries_view;

// Iterator Traits

template <class T>
struct const_iterator_type {
  typedef typename T::const_iterator type;
};
template <class ValueType>
struct const_iterator_type<alps::alea::mcdata<ValueType> > {
  typedef typename std::vector<ValueType>::const_iterator type;
}; 


template <class T>
struct iterator_type {
  typedef typename T::iterator type;
};
template <class ValueType>
struct iterator_type<alps::alea::mcdata<ValueType> > {
  typedef typename std::vector<ValueType>::iterator type;
}; 

// MCTIMESERIES CLASS DEFENITION
template <class ValueType>
class mctimeseries {
public:
  typedef std::size_t size_type;
  typedef ValueType value_type;
  typedef typename average_type<ValueType>::type average_type;
  typedef typename std::vector<ValueType>::iterator iterator;
  typedef typename std::vector<ValueType>::const_iterator const_iterator;

  friend class mctimeseries_view<value_type>;

  // Constructors

  mctimeseries():_timeseries(new std::vector<ValueType>) {}

  mctimeseries(const alps::alea::mcdata<ValueType>& IN):_timeseries(new std::vector<ValueType>(IN.bins())) {}

  mctimeseries(const alps::alea::mctimeseries<ValueType>& IN):_timeseries(new std::vector<ValueType>(IN.timeseries())) {}

  mctimeseries(const alps::alea::mctimeseries_view<ValueType>& IN):_timeseries(new std::vector<ValueType>(IN.begin(), IN.end())) {}

  // debug constructors. can eventually be deleted.
  mctimeseries(const std::vector<ValueType>& timeseries):_timeseries(new std::vector<ValueType>(timeseries)) {}
#ifdef ALPS_HAVE_PYTHON
  mctimeseries(boost::python::object IN);
#endif

  // shallow assign
  void shallow_assign(const mctimeseries<ValueType>& IN) {
    _timeseries = IN._timeseries;
  }

  // Begin / End
  inline const_iterator begin () const {return (*_timeseries).begin();}
  inline const_iterator end () const {return (*_timeseries).end();}

  inline iterator begin() {return (*_timeseries).begin();}
  inline iterator end() {return (*_timeseries).end();}

  std::size_t size() const { 
    return _timeseries->size();
  }

  // std::Vector-like interface
  inline void push_back (value_type IN) {
    (*_timeseries).push_back(IN);
  }

  inline void resize(std::size_t size) {
    (*_timeseries).resize(size);
  }

  // get functions
  inline std::vector<ValueType> timeseries() const {return *_timeseries;}
#ifdef ALPS_HAVE_PYTHON
  boost::python::object timeseries_python() const;
#endif

  void print () const {
    using alps::numeric::operator<<;
    for (const_iterator it = begin(); it != end(); ++it) {
    std::cout << *it;
    }
  }

private:

  boost::shared_ptr< std::vector<ValueType> > _timeseries;

};


// MCTIMESERIES_VIEW CLASS DEFENITION
template <class ValueType>
class mctimeseries_view {
public:
  typedef std::size_t size_type;
  typedef ValueType value_type;
  typedef typename average_type<ValueType>::type average_type;
  typedef typename std::vector<ValueType>::iterator iterator;
  typedef typename std::vector<ValueType>::const_iterator const_iterator;

  // constructors
  mctimeseries_view(const mctimeseries<ValueType>& timeseries): _timeseries(timeseries._timeseries), _front_cutoff(0), _back_cutoff(0) {};

  void cut_head (int cutoff) {
    if (cutoff < 0) cutoff += size();
    _front_cutoff += cutoff;
  }
  void cut_tail (int cutoff) {
    if (cutoff < 0) cutoff += size();
    _back_cutoff += cutoff;
  }

  // begin + end
  inline const_iterator begin () const {return (*_timeseries).begin() + _front_cutoff;}
  inline const_iterator end () const {return (*_timeseries).end() - _back_cutoff;}
 
  std::size_t size() const 
  {
      return _timeseries->size()-_front_cutoff-_back_cutoff;
  }

  // this copies the sub-vector. is there a better way?
  inline std::vector<ValueType> timeseries() const {return std::vector<ValueType>(begin(), end());}
#ifdef ALPS_HAVE_PYTHON
  boost::python::object timeseries_python() const;
#endif

  void print () const {
    using alps::numeric::operator<<;
    for (const_iterator it = begin(); it != end(); ++it) {
    std::cout << *it;
    }
  }

private:
  boost::shared_ptr< std::vector<ValueType> > _timeseries;
  std::size_t _front_cutoff;
  std::size_t _back_cutoff;
};


// RANGE_BEGIN AND RANGE_END
template <class TimeseriesType>
typename const_iterator_type<TimeseriesType>::type range_begin (const TimeseriesType& timeseries) {
  return timeseries.begin();
}
template <class ValueType>
typename std::vector<ValueType>::const_iterator range_begin (const alps::alea::mcdata<ValueType>& timeseries) {
  return timeseries.bins().begin();
}


template <class TimeseriesType>
typename const_iterator_type<TimeseriesType>::type range_end (const TimeseriesType& timeseries) {
  return timeseries.end();
}
template <class ValueType>
typename std::vector<ValueType>::const_iterator range_end (const alps::alea::mcdata<ValueType>& timeseries) {
  return timeseries.bins().end();
}

// Error if not enough measurements are availale
class NotEnoughMeasurementsError : public std::runtime_error {
public:
  NotEnoughMeasurementsError()
   : std::runtime_error("Not enough measurements available.")
   { }
};


  /*
// SIZE -- to be removed -> XX.size()     need a .size() in mcdata first
template <class TimeseriesType>
std::size_t size(const TimeseriesType& timeseries){
  std::size_t OUT = 0;
  for (typename const_iterator_type<TimeseriesType>::type iter = range_begin(timeseries); iter != range_end(timeseries); ++iter)
    ++OUT;
  return OUT;
}
*/
  

// Cut Head / Tail

template <class TimeseriesType>
mctimeseries_view<typename TimeseriesType::value_type > cut_head_distance (const TimeseriesType& timeseries, int cutoff) {
  mctimeseries_view<typename TimeseriesType::value_type > OUT(timeseries);
  OUT.cut_head(cutoff);
  return OUT;
}
template <class TimeseriesType>
mctimeseries_view<typename TimeseriesType::value_type > cut_tail_distance (const TimeseriesType& timeseries, int cutoff) {
  mctimeseries_view<typename TimeseriesType::value_type > OUT(timeseries);
  OUT.cut_tail(cutoff);
  return OUT;
}

template <class TimeseriesType>   
mctimeseries_view<typename TimeseriesType::value_type > cut_head_limit (const TimeseriesType& timeseries, double limit) {  
  mctimeseries_view<typename TimeseriesType::value_type > OUT(timeseries);  
  int cutoff(0);
  limit = limit * *OUT.begin();  
  std::find_if(OUT.begin(), OUT.end(), ( ++boost::lambda::var(cutoff), boost::lambda::_1 <= limit ));  
  OUT.cut_head(cutoff);  
  return OUT;  
}
template <class TimeseriesType>   
mctimeseries_view<typename TimeseriesType::value_type > cut_tail_limit (const TimeseriesType& timeseries, double limit) {  
  mctimeseries_view<typename TimeseriesType::value_type > OUT(timeseries);  
  int cutoff(0);
  limit = limit * *OUT.begin();  
  std::find_if(OUT.begin(), OUT.end(), ( ++boost::lambda::var(cutoff), boost::lambda::_1 <= limit ));  
  cutoff = OUT.size() - cutoff;
  OUT.cut_tail(cutoff);  
  return OUT;  
}

// MEAN
template <class TimeseriesType>
typename average_type< typename TimeseriesType::value_type >::type mean(const TimeseriesType& timeseries) {
  typedef typename average_type< typename TimeseriesType::value_type >::type return_type;
  using boost::numeric::operators::operator/;
  using boost::numeric::operators::operator+;

  if ( range_begin(timeseries) == range_end(timeseries) ) boost::throw_exception(NotEnoughMeasurementsError());

  return_type OUT;
  resize_same_as(OUT, *range_begin(timeseries) );
  set_zero(OUT);

  for (typename const_iterator_type<TimeseriesType>::type iter = range_begin(timeseries); iter != range_end(timeseries); ++iter)
    OUT = OUT + *iter;

  return OUT / double(size(timeseries));
}


// VARIANCE
template <class TimeseriesType>
typename average_type< typename TimeseriesType::value_type >::type variance(const TimeseriesType& timeseries) {  
  typedef typename average_type< typename TimeseriesType::value_type >::type return_type;
  using boost::numeric::operators::operator-;
  using boost::numeric::operators::operator+;
  using boost::numeric::operators::operator/;
  using std::pow;
  using alps::numeric::pow;

  if (size(timeseries) < 2) boost::throw_exception(NotEnoughMeasurementsError());

  return_type _mean = mean(timeseries);
  return_type OUT;
  resize_same_as(OUT, *range_begin(timeseries) );
  set_zero(OUT);

  for (typename const_iterator_type<TimeseriesType>::type iter = range_begin(timeseries); iter != range_end(timeseries); ++iter) {
    OUT = OUT + pow(*iter-_mean, 2.);
  }

  return OUT / double(size(timeseries) - 1);
}


// AUTOCORRELATION

template <class TimeseriesType>
mctimeseries< typename average_type< typename TimeseriesType::value_type >::type > autocorrelation_distance(const TimeseriesType& timeseries, int up_to) {
  typedef typename average_type<typename TimeseriesType::value_type >::type average_type;
  using boost::numeric::operators::operator-;
  using boost::numeric::operators::operator*;
  using boost::numeric::operators::operator/;
  using boost::numeric::operators::operator+;

  std::size_t _size = size(timeseries);
  average_type _mean = alps::alea::mean(timeseries);
  average_type _variance = alps::alea::variance(timeseries);
  mctimeseries< average_type > OUT;

  if (_size < 2) boost::throw_exception(NotEnoughMeasurementsError());
  if (up_to < 0) up_to += _size;

  average_type tmp;
  resize_same_as(tmp, *range_begin(timeseries) );

  for (std::size_t i = 1; i <= up_to; ++i) {
    if (i == _size) {std::cout << "  Warning: Autocorrelation fully calculated with a size of " << i-1 << " !\n"; break;}
    set_zero(tmp);
    for (typename const_iterator_type<TimeseriesType>::type iter = range_begin(timeseries); iter != range_end(timeseries) - i ; ++iter) {
      tmp = tmp + (*iter - _mean) * ( *(iter + i) - _mean);
    }
    tmp = tmp / ( _variance * (_size - i) );
    OUT.push_back(tmp);
  }
  return OUT;
}


template <class TimeseriesType>
mctimeseries< typename average_type< typename TimeseriesType::value_type >::type > autocorrelation_limit(const TimeseriesType& timeseries, double limit) {
  typedef typename average_type< typename TimeseriesType::value_type >::type average_type;
  using boost::numeric::operators::operator-;
  using boost::numeric::operators::operator*;
  using boost::numeric::operators::operator/;
  using boost::numeric::operators::operator+;

  std::size_t _size = size(timeseries);
  average_type _mean = mean(timeseries);
  average_type _variance = variance(timeseries);
  mctimeseries< average_type > OUT;

  if (_size < 2) boost::throw_exception(NotEnoughMeasurementsError());

  average_type tmp;
  resize_same_as(tmp, *range_begin(timeseries) );
  std::size_t i = 1;

  while (true) {
    if (i == _size) {std::cout << "  Warning: Autocorrelation fully calculated with a size of " << _size - 1 << " !\n"; break;}
    set_zero(tmp);
    for (typename const_iterator_type<TimeseriesType>::type iter = range_begin(timeseries); iter != range_end(timeseries) - i ; ++iter) {
      tmp = tmp + (*iter - _mean) * ( *(iter + i) - _mean);
    }
    tmp = tmp / ( _variance * (_size - i) );
    OUT.push_back(tmp);
    if ( alps::numeric::at_least_one( tmp, limit * (*range_begin(OUT)) , (boost::lambda::_1 < boost::lambda::_2) )  ) break;
    ++i;
  }
  return OUT;
}


// EXPONENTIAL AUTOCORRELATION TIME

template <class TimeseriesType>
std::pair<typename average_type<typename TimeseriesType::value_type >::type, typename average_type<typename TimeseriesType::value_type >::type>
          exponential_autocorrelation_time_distance (const TimeseriesType& autocorrelation, int from, int to) {
  typedef typename average_type<typename TimeseriesType::value_type >::type average_type;
  using std::exp;

  if (from < 0) from = from + size(autocorrelation);
  if (to < 0) to = to + size(autocorrelation);

  mctimeseries_view<average_type> autocorrelation_view = cut_head_distance(cut_tail_distance(autocorrelation, size(autocorrelation) - to), from - 1);

  std::pair<average_type, average_type> OUT( alps::numeric::exponential_timeseries_fit(autocorrelation_view.begin(), autocorrelation_view.end()) );
  OUT.first *= exp((-1.) * OUT.second * (from - 1));

  return OUT;
}


template <class TimeseriesType>
std::pair<typename average_type<typename TimeseriesType::value_type >::type, typename average_type<typename TimeseriesType::value_type >::type>
          exponential_autocorrelation_time_limit (const TimeseriesType& autocorrelation, double max, double min) {
  typedef typename average_type<typename TimeseriesType::value_type >::type average_type;

  average_type first = *(autocorrelation.begin());
  max *= first;
  min *= first;

  std::size_t from_int(0);
  std::size_t to_int(0);

  std::find_if(autocorrelation.begin(), autocorrelation.end(), ( ++boost::lambda::var(from_int), boost::lambda::_1 <= max ));
  std::find_if(autocorrelation.begin(), autocorrelation.end(), ( ++boost::lambda::var(to_int), boost::lambda::_1 <= min ));

  if ( (to_int - 1) < from_int) std::cout << "Warning: Invalid Range! If you want to fit a positive exponential, exchange min and max.\n";

  return exponential_autocorrelation_time_distance (autocorrelation, from_int, to_int - 1);

}


// Integrated Autocorrelation Time
template <class TimeseriesType>
typename average_type< typename TimeseriesType::value_type >::type integrated_autocorrelation_time(
        const TimeseriesType&                                                       autocorrelation,
        const std::pair< typename average_type<typename TimeseriesType::value_type>::type ,  typename average_type<typename TimeseriesType::value_type>::type >&
                                                                                    tau  )
{

  typedef typename average_type< typename TimeseriesType::value_type >::type return_type;

  return_type OUT = std::accumulate(autocorrelation.begin(), autocorrelation.end(), 0.);

  OUT -= (tau.first / tau.second) * std::exp(tau.second * (size(autocorrelation) + 0.5));

  return OUT;
}


// Error

  //call methods
struct uncorrelated_selector {} uncorrelated;
struct binning_selector {} binning;


template <class TimeseriesType>
typename average_type< typename TimeseriesType::value_type >::type error (const TimeseriesType& timeseries, const uncorrelated_selector& selector = alps::alea::uncorrelated) {
  using std::sqrt;
  using alps::numeric::sqrt;
  using boost::numeric::operators::operator/;

  return sqrt( variance(timeseries) / double(size(timeseries)) );
}


template <class TimeseriesType>
typename average_type< typename TimeseriesType::value_type >::type error (const TimeseriesType& timeseries, const binning_selector& selector) {
  typedef typename TimeseriesType::value_type value_type;
  using boost::numeric::operators::operator/;

  alps::SimpleBinning<typename vector2valarray_type<value_type>::type> _binning;

  for (typename const_iterator_type<TimeseriesType>::type iter = range_begin(timeseries); iter != range_end(timeseries); ++iter) {
    _binning << alps::numeric::vector2valarray(*iter);
  };

  return alps::numeric::valarray2vector(_binning.error() );
}


// error() interface for python export
template <class TimeseriesType>
inline typename average_type< typename TimeseriesType::value_type >::type uncorrelated_error (const TimeseriesType& timeseries) {return error(timeseries, uncorrelated);}

template <class TimeseriesType>
inline typename average_type< typename TimeseriesType::value_type >::type binning_error (const TimeseriesType& timeseries) {return error(timeseries, binning);}



// Running Mean
template <class TimeseriesType>
mctimeseries< typename average_type<typename TimeseriesType::value_type>::type > running_mean(const TimeseriesType& timeseries) {
  typedef typename average_type<typename TimeseriesType::value_type>::type average_type;
  typedef mctimeseries< average_type > return_type;
  using boost::numeric::operators::operator/;

  return_type _running_mean;
  _running_mean.resize(size(timeseries) );

  std::partial_sum(range_begin(timeseries), range_end(timeseries), _running_mean.begin(), alps::numeric::plus<average_type, average_type, average_type>() );

  std::size_t count = 0;
  for (typename return_type::iterator iter = _running_mean.begin(); iter != _running_mean.end(); ++iter)
    *iter = *iter / ++count;

  return _running_mean;
}

//  Reverse Running Mean
template <class TimeseriesType>
mctimeseries< typename average_type<typename TimeseriesType::value_type>::type > reverse_running_mean(const TimeseriesType& timeseries) {
  typedef typename average_type<typename TimeseriesType::value_type>::type average_type;
  typedef mctimeseries< average_type > return_type;
  using boost::numeric::operators::operator/;

  mctimeseries<average_type> _reverse_running_mean;
  _reverse_running_mean.resize(size(timeseries) );

  std::partial_sum(static_cast <std::reverse_iterator<typename const_iterator_type<TimeseriesType>::type> > (range_end(timeseries)),
                   static_cast <std::reverse_iterator<typename const_iterator_type<TimeseriesType>::type> > (range_begin(timeseries)),
                   static_cast <std::reverse_iterator<typename iterator_type<TimeseriesType>::type> > (_reverse_running_mean.end() ), alps::numeric::plus<average_type, average_type, average_type>() );

  std::size_t count = size(timeseries);
  for (typename iterator_type<TimeseriesType>::type iter = _reverse_running_mean.begin(); iter != _reverse_running_mean.end(); ++iter)
    *iter = *iter / count--;

  return _reverse_running_mean;
}


// define distance/limit boost::parameter interface 

#define ALPS_ALEA_IMPL_DISTANCE_LIMIT_FUNCTION(name) \
 name (const TimeseriesType& timeseries, ArgumentPack const& arg) {  \
\
  if ( arg[_distance|None] != None) {  \
    return name## _distance(timeseries, arg[_distance|None]);  \
  } \
  if ( arg[_limit|None] != None) { \
    return name## _limit(timeseries, arg[_limit|None]);  \
  } \
\
  boost::throw_exception(std::runtime_error("Either distance or limit must be specified!")); \
}

template <class TimeseriesType, class ArgumentPack>
mctimeseries< typename average_type< typename TimeseriesType::value_type >::type >
  ALPS_ALEA_IMPL_DISTANCE_LIMIT_FUNCTION(autocorrelation)

template <class TimeseriesType, class ArgumentPack>
mctimeseries_view<typename TimeseriesType::value_type >
  ALPS_ALEA_IMPL_DISTANCE_LIMIT_FUNCTION(cut_head)

template <class TimeseriesType, class ArgumentPack>
mctimeseries_view<typename TimeseriesType::value_type >
  ALPS_ALEA_IMPL_DISTANCE_LIMIT_FUNCTION(cut_tail)

#undef ALPS_ALEA_IMPL_DISTANCE_LIMIT_FUNCTION


// define from-to/min-max boost::parameter interface 
// currently this is only needed for exponential_autocorrelation_time, but maybe future functions will need this too.

#define ALPS_ALEA_IMPL_FROM_TO_MIN_MAX_FUNCTION(name) \
 name (const TimeseriesType& timeseries, ArgumentPack1 const& arg1, ArgumentPack2 const& arg2) {  \
\
  boost::parameter::parameters< \
    boost::parameter::optional<tag::from, boost::is_convertible<tag::from::_,int> > \
    , boost::parameter::optional<tag::to, boost::is_convertible<tag::to::_,int> >  \
    , boost::parameter::optional<tag::min, boost::is_convertible<tag::min::_,double> > \
    , boost::parameter::optional<tag::max, boost::is_convertible<tag::max::_,double> > \
  > spec; \
\
  if ( (spec(arg1,arg2)[_from|None] != None) && (spec(arg1,arg2)[_to|None] != None) ) {  \
    return name## _distance(timeseries, spec(arg1,arg2)[_from|None], spec(arg1,arg2)[_to|None]);  \
  } \
  if ( (spec(arg1,arg2)[_max|None] != None) && (spec(arg1,arg2)[_min|None] != None) ) {  \
    return name## _limit(timeseries, spec(arg1,arg2)[_max|None], spec(arg1,arg2)[_min|None]);  \
  } \
\
  boost::throw_exception(std::runtime_error("Either max, min or from, to must be specified!")); \
}

template <class TimeseriesType, class ArgumentPack1, class ArgumentPack2>
std::pair<typename average_type<typename TimeseriesType::value_type >::type, typename average_type<typename TimeseriesType::value_type >::type>
  ALPS_ALEA_IMPL_FROM_TO_MIN_MAX_FUNCTION(exponential_autocorrelation_time)

#undef ALPS_ALEA_IMPL_FROM_TO_MIN_MAX_FUNCTION


// OStream
#define ALPS_MCANALYZE_IMPLEMENT_OSTREAM(timeseries_type)                                                              \
template <typename ValueType>                                                                                         \
std::ostream& operator<<(std::ostream & out, timeseries_type <ValueType> const & timeseries) {                                  \
  using alps::numeric::operator<<;  \
  for (typename timeseries_type <ValueType>:: const_iterator it = timeseries.begin(); it != timeseries.end(); ++it) {  \
    out << *it;             \
  }  \
  return out;                                                                                                                 \
}
ALPS_MCANALYZE_IMPLEMENT_OSTREAM(mctimeseries)
ALPS_MCANALYZE_IMPLEMENT_OSTREAM(mctimeseries_view)
#undef ALPS_MCANALYZE_IMPLEMENT_OSTREAM


} // ending namespace alea
} // ending namespace alps


#endif



