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
  typedef size_t size_type;
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

  size_t size() {
    size_t OUT = 0;
    for (const_iterator iter = begin(); iter != end(); ++iter)
      ++OUT;
    return OUT;
  }

  // std::Vector-like interface
  inline void push_back (value_type IN) {
    (*_timeseries).push_back(IN);
  }

  inline void resize(size_t size) {
    (*_timeseries).resize(size);
  }

  // get functions
  inline std::vector<ValueType> timeseries() const {return *_timeseries;}
#ifdef ALPS_HAVE_PYTHON
  boost::python::object timeseries_python() const;
#endif

  // DEBUG print
  void print () const {
    std::copy(begin(), end(), std::ostream_iterator<ValueType>(std::cout, " "));
  }

private:

  boost::shared_ptr< std::vector<ValueType> > _timeseries;

};


// MCTIMESERIES_VIEW CLASS DEFENITION
template <class ValueType>
class mctimeseries_view {
public:
  typedef size_t size_type;
  typedef ValueType value_type;
  typedef typename average_type<ValueType>::type average_type;
  typedef typename std::vector<ValueType>::iterator iterator;
  typedef typename std::vector<ValueType>::const_iterator const_iterator;

  // constructors
  mctimeseries_view(const mctimeseries<ValueType>& timeseries): _timeseries(timeseries._timeseries), _front_cutoff(0), _back_cutoff(0) {};

  void cut_head (size_t cutoff) {_front_cutoff += cutoff;}
  void cut_tail (size_t cutoff) {_back_cutoff += cutoff;}

  // begin + end
  inline const_iterator begin () const {return (*_timeseries).begin() + _front_cutoff;}
  inline const_iterator end () const {return (*_timeseries).end() - _back_cutoff;}
 
  size_t size() {
    size_t OUT = 0;
    for (const_iterator iter = begin(); iter != end(); ++iter)
      ++OUT;
    return OUT;
  }

  // this copies the sub-vector. is there a better way?
  inline std::vector<ValueType> timeseries() const {return std::vector<ValueType>(begin(), end());}

#ifdef ALPS_HAVE_PYTHON
  boost::python::object timeseries_python() const;
#endif

private:
  boost::shared_ptr< std::vector<ValueType> > _timeseries;
  size_t _front_cutoff;
  size_t _back_cutoff;
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


// SIZE -- to be removed -> XX.size()     need a .size() in mcdata first
template <class TimeseriesType>
size_t size(const TimeseriesType& timeseries){
  size_t OUT = 0;
  for (typename const_iterator_type<TimeseriesType>::type iter = range_begin(timeseries); iter != range_end(timeseries); ++iter)
    ++OUT;
  return OUT;
}


// Cut Head / Tail
#define ALPS_ALEA_CUT_HEAD_TAIL_IMPL(name)              \
template <class TimeseriesType, class ArgumentPack>   \
mctimeseries_view<typename TimeseriesType::value_type > name (const TimeseriesType& timeseries, ArgumentPack const& arg) {  \
  mctimeseries_view<typename TimeseriesType::value_type > OUT(timeseries);  \
  size_t cutoff(0);  \
  if (arg[_distance|0]) cutoff = arg[_distance|0];  \
  if (arg[_limit|0]) {  \
    double limit = arg[_limit|0] * *timeseries.begin();  \
    std::find_if(timeseries.begin(), timeseries.end(), ( ++boost::lambda::var(cutoff), boost::lambda::_1 <= limit ));  \
    cutoff = alps::alea::size(timeseries) - cutoff;  \
  }  \
  OUT. name (cutoff);  \
  return OUT;  \
}

ALPS_ALEA_CUT_HEAD_TAIL_IMPL(cut_head)
ALPS_ALEA_CUT_HEAD_TAIL_IMPL(cut_tail)

#undef ALPS_ALEA_CUT_HEAD_TAIL_IMPL


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

  return OUT / double(alps::alea::size(timeseries));
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

  if (alps::alea::size(timeseries) < 2) boost::throw_exception(NotEnoughMeasurementsError());

  return_type _mean = mean(timeseries);
  return_type OUT;
  resize_same_as(OUT, *range_begin(timeseries) );
  set_zero(OUT);

  for (typename const_iterator_type<TimeseriesType>::type iter = range_begin(timeseries); iter != range_end(timeseries); ++iter) {
    OUT = OUT + pow(*iter-_mean, 2.);
  }

  return OUT / double(alps::alea::size(timeseries) - 1);
}


// AUTOCORRELATION

template <class TimeseriesType>
mctimeseries< typename average_type< typename TimeseriesType::value_type >::type > autocorrelation_distance(const TimeseriesType& timeseries, int up_to) {
  typedef typename average_type<typename TimeseriesType::value_type >::type average_type;
  using boost::numeric::operators::operator-;
  using boost::numeric::operators::operator*;
  using boost::numeric::operators::operator/;
  using boost::numeric::operators::operator+;

  size_t _size = alps::alea::size(timeseries);
  average_type _mean = alps::alea::mean(timeseries);
  average_type _variance = alps::alea::variance(timeseries);
  mctimeseries< average_type > OUT;

  if (_size < 2) boost::throw_exception(NotEnoughMeasurementsError());
  if (up_to < 0) up_to += _size;

  average_type tmp;
  resize_same_as(tmp, *range_begin(timeseries) );

  for (size_t i = 1; i <= up_to; ++i) {
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
mctimeseries< typename average_type< typename TimeseriesType::value_type >::type > autocorrelation_limit(const TimeseriesType& timeseries, double percentage) {
  typedef typename average_type< typename TimeseriesType::value_type >::type average_type;
  using boost::numeric::operators::operator-;
  using boost::numeric::operators::operator*;
  using boost::numeric::operators::operator/;
  using boost::numeric::operators::operator+;

  size_t _size = alps::alea::size(timeseries);
  average_type _mean = mean(timeseries);
  average_type _variance = variance(timeseries);
  mctimeseries< average_type > OUT;

  if (_size < 2) boost::throw_exception(NotEnoughMeasurementsError());

  average_type tmp;
  resize_same_as(tmp, *range_begin(timeseries) );
  size_t i = 1;

  while (true) {
    if (i == _size) {std::cout << "  Warning: Autocorrelation fully calculated with a size of " << _size - 1 << " !\n"; break;}
    set_zero(tmp);
    for (typename const_iterator_type<TimeseriesType>::type iter = range_begin(timeseries); iter != range_end(timeseries) - i ; ++iter) {
      tmp = tmp + (*iter - _mean) * ( *(iter + i) - _mean);
    }
    tmp = tmp / ( _variance * (_size - i) );
    OUT.push_back(tmp);
    if ( alps::numeric::at_least_one( tmp, percentage * (*range_begin(OUT)) , (boost::lambda::_1 < boost::lambda::_2) )  ) break;
    ++i;
  }
  return OUT;
}

template <class TimeseriesType, class ArgumentPack>
mctimeseries< typename average_type<typename TimeseriesType::value_type >::type >
      autocorrelation (const TimeseriesType& timeseries, ArgumentPack const& arg) {

  if (arg[_distance|0]) {;
    return autocorrelation_distance(timeseries, arg[_distance|0]);
  }

  if (arg[_limit|0]) {
    return autocorrelation_limit(timeseries, arg[_limit|0]);
  }
  std::cout << "nothing\n"; // limit = 0 problem!

}


// CORRELATION TIME

template <class TimeseriesType, class ArgumentPack>
std::pair<typename average_type<typename TimeseriesType::value_type >::type, typename average_type<typename TimeseriesType::value_type >::type>
          exponential_autocorrelation_time (const TimeseriesType& autocorrelation, ArgumentPack const& arg) {

  typedef typename average_type<typename TimeseriesType::value_type >::type average_type;

  int from(0);
  int to(0);

  if (arg[_from|0] != arg[_to|0]) {
    from = arg[_from|0];
    to = arg[_to|0];

    if (arg[_from|0] < 0) from = arg[_from|0] + alps::alea::size(autocorrelation);
    if (arg[_to|0] < 0) to = arg[_to|0] + alps::alea::size(autocorrelation);
  }

  if (arg[_min|0] != arg[_max|0]) {
    double min = arg[_min|0];
    double max = arg[_max|0];

    average_type first = *(autocorrelation.begin());
    min *= first;
    max *= first;

    size_t from_int(0);
    size_t to_int(0);

    std::find_if(autocorrelation.begin(), autocorrelation.end(), ( ++boost::lambda::var(from_int), boost::lambda::_1 <= max ));
    std::find_if(autocorrelation.begin(), autocorrelation.end(), ( ++boost::lambda::var(to_int), boost::lambda::_1 <= min ));

    if ( (to_int - 1) < from_int) std::cout << "Warning: Invalid Range!\n";
    from = from_int;
    to = to_int - 1;
  }


  using std::exp;

  mctimeseries_view<average_type> autocorrelation_view = cut_head(cut_tail(autocorrelation, _distance = alps::alea::size(autocorrelation) - to), _distance = from - 1);

  std::pair<average_type, average_type> OUT( alps::numeric::exponential_timeseries_fit(autocorrelation_view.begin(), autocorrelation_view.end()) );
  OUT.first *= exp((-1.) * OUT.second * (from - 1));

  return OUT;
}



// Integrated Correlation Time
template <class TimeseriesType>
typename average_type< typename TimeseriesType::value_type >::type integrated_autocorrelation_time(
        const TimeseriesType&                                                       autocorrelation,
        const std::pair< typename average_type<typename TimeseriesType::value_type>::type ,  typename average_type<typename TimeseriesType::value_type>::type >&
                                                                                    tau  )
{

  typedef typename average_type< typename TimeseriesType::value_type >::type return_type;

  return_type OUT = std::accumulate(autocorrelation.begin(), autocorrelation.end(), 0.);

  OUT -= (tau.first / tau.second) * std::exp(tau.second * (alps::alea::size(autocorrelation) + 0.5));

  return OUT;
}


// Error

  //call methods
struct uncorrelated_selector {} uncorrelated;
struct binning_selector {} binning;



template <class TimeseriesType>
typename average_type< typename TimeseriesType::value_type >::type error (const TimeseriesType& timeseries, const uncorrelated_selector& selector) {
  using std::sqrt;
  using alps::numeric::sqrt;
  using boost::numeric::operators::operator/;

  return sqrt( variance(timeseries) / double(alps::alea::size(timeseries)) );
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


// second error() interface for python export and/or use in C++
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
  _running_mean.resize( alps::alea::size(timeseries) );

  std::partial_sum(range_begin(timeseries), range_end(timeseries), _running_mean.begin(), alps::numeric::plus<average_type, average_type, average_type>() );

  size_t count = 0;
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
  _reverse_running_mean.resize( alps::alea::size(timeseries) );

  std::partial_sum(static_cast <std::reverse_iterator<typename const_iterator_type<TimeseriesType>::type> > (range_end(timeseries)),
                   static_cast <std::reverse_iterator<typename const_iterator_type<TimeseriesType>::type> > (range_begin(timeseries)),
                   static_cast <std::reverse_iterator<typename iterator_type<TimeseriesType>::type> > (_reverse_running_mean.end() ), alps::numeric::plus<average_type, average_type, average_type>() );

  size_t count = alps::alea::size(timeseries);
  for (typename iterator_type<TimeseriesType>::type iter = _reverse_running_mean.begin(); iter != _reverse_running_mean.end(); ++iter)
    *iter = *iter / count--;

  return _reverse_running_mean;
}


// OStream
#define ALPS_MCANALYZE_IMPLEMENT_OSTREAM(timeseries_type)                                                              \
template <typename ValueType>                                                                                         \
std::ostream& operator<<(std::ostream & out, timeseries_type <ValueType> const & timeseries) {                                  \
  std::copy(range_begin(timeseries), range_end(timeseries), std::ostream_iterator<ValueType>(out, " "));             \
  return out;                                                                                                                 \
}

ALPS_MCANALYZE_IMPLEMENT_OSTREAM(mctimeseries)
ALPS_MCANALYZE_IMPLEMENT_OSTREAM(mctimeseries_view)

#undef ALPS_MCANALYZE_IMPLEMENT_OSTREAM

} // ending namespace alea
} // ending namespace alps


#endif



