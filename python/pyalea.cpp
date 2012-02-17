/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
*                            Lukas Gamper <gamperl@gmail.com>,
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

/* $Id: pyalea.cpp 3520 2010-04-09 16:49:53Z tamama $ */

#define PY_ARRAY_UNIQUE_SYMBOL pyalea_PyArrayHandle

#include <alps/python/make_copy.hpp>
#include <alps/python/numpy_array.hpp>
#include <alps/alea/detailedbinning.h>
#include <alps/numeric/vector_functions.hpp>
#include <alps/python/save_observable_to_hdf5.hpp>
#include <alps/random.h>
#include <alps/alea/mcdata.hpp>
#include <alps/alea/mcanalyze.hpp>
#include <alps/alea/value_with_error.hpp>
#include <alps/hdf5.hpp>

#include <boost/python.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <numpy/arrayobject.h>

#include <vector>
#include <string>

using namespace boost::python;

namespace alps { 
  namespace alea {


    template<typename T>
    class WrappedValarrayObservable
    {
      typedef typename T::value_type::value_type element_type;
    public:
      WrappedValarrayObservable(const std::string& name, int s=0)
      : obs(name,s)
      {}
      
      void operator<<(const boost::python::object& arr)
      {
          obs << alps::python::numpy::convert2valarray<element_type>(arr);
      }
      
      std::string representation() const
      {
          return obs.representation();
      }
      
      boost::python::numeric::array mean() const 
      {
          return alps::python::numpy::convert2numpy(obs.mean());
      }
      
      boost::python::numeric::array error() const 
      {
          return alps::python::numpy::convert2numpy(obs.error());
      }
      
      boost::python::numeric::array tau() const 
      {
          return alps::python::numpy::convert2numpy(obs.tau());
      }

     boost::python::numeric::array variance() const 
      {
          return alps::python::numpy::convert2numpy(obs.variance());
      }
      
      void save(std::string const & filename) const {
          hdf5::archive ar(filename, hdf5::archive::WRITE);
          ar << make_pvp("/simulation/results/"+obs.representation(), obs);
      }

      typename T::count_type count() const 
      {
          return obs.count();
      }

      typename T::convergence_type converged_errors() const
      {
          return obs.converged_errors();
      }
      
    private:
      T obs;
      
    };


    template <class T>
    value_with_error<T>::value_with_error(boost::python::object const & mean_nparray, boost::python::object const & error_nparray):
        _mean(alps::python::numpy::convert2vector<T>(mean_nparray) ),
        _error(alps::python::numpy::convert2vector<T>(error_nparray) ) {}

    //boost::python::object value_with_error::mean_nparray() const;
    //boost::python::object value_with_error::error_nparray() const;

    template <typename T>
    boost::python::str print_value_with_error(alps::alea::value_with_error<T> const & self) {
      return boost::python::str(boost::python::str(self.mean()) + " +/- " + boost::python::str(self.error()));
    }


    #define ALPS_ALEA_FUNCTION_NUMPY_WRAPPER(function_name) \
    template <class T> \
    boost::python::numeric::array function_name## _wrapper ( const T& arg1 ) { \
      return alps::python::numpy::convert(function_name( arg1 )) ;  \
    } 

    ALPS_ALEA_FUNCTION_NUMPY_WRAPPER(mean)
    ALPS_ALEA_FUNCTION_NUMPY_WRAPPER(variance)

    #undef ALPS_ALEA_FUNCTION_NUMPY_WRAPPER



    template <class ValueType>
    mctimeseries<ValueType>::mctimeseries (boost::python::object IN):_timeseries(new std::vector<ValueType>( alps::python::numpy::convert2vector<ValueType>(IN) )) {}

    template <class ValueType>
    boost::python::object mctimeseries<ValueType>::timeseries_python() const {return alps::python::numpy::convert(timeseries());}

    template <class ValueType>
    boost::python::object mctimeseries_view<ValueType>::timeseries_python() const {return alps::python::numpy::convert(timeseries());}


  } // ending namespace alea
} // ending namespace alps

using namespace alps::alea;
using namespace alps::numeric;

const char constructor_docstring[] = 
"The constructor takes two arguments: a string with the name of the observable "
"and optionally a second integer argument specifying the number of bins to be "
"stored.";

const char timeseries_constructor_docstring[] = 
"The constructor takes two arguments: a string with the name of the observable "
"and optionally a second integer argument specifying the number of entries per " 
"bin in the time series.";

const char observable_docstring[] =
"This class is an ALPS observable class to record results of Monte Carlo "
"measurements and evaluate mean values, error, and autocorrelations.";

const char timeseries_observable_docstring[] =
"This class is an ALPS observable class to record results of Monte Carlo "
"measurements and evaluate mean values, error, and autocorrelations. "
"It records a full binned time series of measurements, where the number of "
"elements per bin can be specified.";

const char shift_docstring[] =
"New measurements are added using the left shift operator <<.";

const char save_docstring[] =
"Save the obseravble into the HDF5 file specified as the argument.";

const char mean_docstring[] =
"the mean value of all measurements recorded.";

const char error_docstring[] =
"the error of all measurements recorded.";

const char tau_docstring[] =
"the autocorrelation time estimate of the recorded measurements.";

const char variance_docstring[] =
"the variance of all measurements recorded.";

const char count_docstring[] =
"the number of measurements recorded.";

const char converged_errors_docstring[] =
" (0 -- data converged ; 1 -- data maybe converged ; 2 -- data not converged) ";


BOOST_PYTHON_MODULE(pyalea_c) {
#define ALPS_PY_EXPORT_VECTOROBSERVABLE(class_name, class_docstring, init_docstring)                                            \
  class_<WrappedValarrayObservable< alps:: class_name > >(                                                                      \
       #class_name, class_docstring, init<std::string, optional<int> >(init_docstring))                                         \
    .def("__repr__", &WrappedValarrayObservable< alps:: class_name >::representation)                                           \
    .def("__deepcopy__",  &alps::python::make_copy<WrappedValarrayObservable< alps::class_name > >)                             \
    .def("__lshift__", &WrappedValarrayObservable< alps::class_name >::operator<<,shift_docstring)                              \
    .def("save", &WrappedValarrayObservable< alps::class_name >::save,save_docstring)                                           \
    .add_property("mean", &WrappedValarrayObservable< alps::class_name >::mean,mean_docstring)                                  \
    .add_property("error", &WrappedValarrayObservable< alps::class_name >::error,error_docstring)                               \
    .add_property("tau", &WrappedValarrayObservable< alps::class_name >::tau,tau_docstring)                                     \
    .add_property("variance", &WrappedValarrayObservable< alps::class_name >::variance,variance_docstring)                      \
    .add_property("count", &WrappedValarrayObservable< alps::class_name >::count,count_docstring)                               \
    .add_property("converged_errors", &WrappedValarrayObservable< alps::class_name >::converged_errors,converged_errors_docstring) \
    ;                                                                                                                           

ALPS_PY_EXPORT_VECTOROBSERVABLE(RealVectorObservable,observable_docstring,constructor_docstring)
ALPS_PY_EXPORT_VECTOROBSERVABLE(RealVectorTimeSeriesObservable,timeseries_observable_docstring,timeseries_constructor_docstring)
#undef ALPS_PY_EXPORT_VECTOROBSERVABLE
    
#define ALPS_PY_EXPORT_SIMPLEOBSERVABLE(class_name, class_docstring, init_docstring)                                                \
  class_< alps:: class_name >(#class_name, class_docstring, init<std::string, optional<int> >(init_docstring))                      \
    .def("__deepcopy__",  &alps::python::make_copy<alps:: class_name >)                                                             \
    .def("__repr__", &alps:: class_name ::representation)                                                                           \
    .def("__lshift__", &alps:: class_name ::operator<<,shift_docstring)                                                             \
    .def("save", &alps::python::save_observable_to_hdf5<alps:: class_name >,save_docstring)                                         \
    .add_property("mean", &alps:: class_name ::mean,mean_docstring)                                                                 \
    .add_property("error", static_cast<alps:: class_name ::result_type(alps:: class_name ::*)() const>(&alps:: class_name ::error),error_docstring) \
    .add_property("tau",&alps:: class_name ::tau,tau_docstring)                                                                     \
    .add_property("variance",&alps:: class_name ::variance,variance_docstring)                                                      \
    .add_property("count",&alps:: class_name ::count,count_docstring)                                                               \
    .add_property("converged_errors", &alps:: class_name ::converged_errors,converged_errors_docstring)                             \
    ;                                                                                                                               \
       
ALPS_PY_EXPORT_SIMPLEOBSERVABLE(RealObservable,observable_docstring,timeseries_constructor_docstring)
ALPS_PY_EXPORT_SIMPLEOBSERVABLE(RealTimeSeriesObservable,timeseries_observable_docstring,timeseries_constructor_docstring)

#undef ALPS_PY_EXPORT_SIMPLEOBSERVABLE



// mcanalyze export

#define QUOTEME(x) #x

#define ALPS_MCANALYZE_EXPORT_MCTIMESERIES_CLASSES(type, name)                                                                  \
  class_<alps::alea::mctimeseries< type > >( QUOTEME(name) )                                                                       \
    .def(init<boost::python::object>())                                                                                 \
    .def(init<alps::alea::mcdata< type > >())                                                                                                 \
    .def("timeseries", &alps::alea::mctimeseries< type >::timeseries_python)                                                          \
  ;                                                                                                                                     \
                                                                                                                                      \
  class_<alps::alea::mctimeseries_view< type > >( QUOTEME(name##View), init< alps::alea::mctimeseries< type > >())               \
    .def(init< alps::alea::mctimeseries_view< type > >())                                                                        \
    .def("timeseries", &alps::alea::mctimeseries_view< type >::timeseries_python)                                                          \
  ; 


#define ALPS_MCANALYZE_EXPORT_HELPER(templateparms, function_name_py, function_name_c)                                       \
    def( QUOTEME ( function_name_py ), function_name_c templateparms);

#define ALPS_MCANALYZE_EXPORT_SCALAR_AND_VECTOR(container_type, function_name_py, function_name_c)                           \
    ALPS_MCANALYZE_EXPORT_HELPER( < container_type < double > > , function_name_py, function_name_c )                        \
    ALPS_MCANALYZE_EXPORT_HELPER( < container_type < std::vector < double > > > , function_name_py, function_name_c )                                        
//    ALPS_MCANALYZE_EXPORT_HELPER( < container_type < int > > , function_name_py, function_name_c )                         \
    ALPS_MCANALYZE_EXPORT_HELPER( < container_type < std::vector < int > > > , function_name_py, function_name_c )                                        

#define ALPS_MCANALYZE_EXPORT_SCALAR_ONLY(container_type, function_name_py, function_name_c)                                 \
    ALPS_MCANALYZE_EXPORT_HELPER( < container_type < double > > , function_name_py, function_name_c )  

#define ALPS_MCANALYZE_EXPORT_VECTOR_ONLY(container_type, function_name_py, function_name_c)                                 \
    ALPS_MCANALYZE_EXPORT_HELPER( < container_type < std::vector < double > > > , function_name_py, function_name_c )    

#define ALPS_MCANALYZE_EXPORT_VALUETYPE_FUNCTION(function_name_py, function_name_c)                                          \
    ALPS_MCANALYZE_EXPORT_HELPER( < double > , function_name_py, function_name_c )                                           \
    ALPS_MCANALYZE_EXPORT_HELPER( < std::vector < double > > , function_name_py, function_name_c )
//    ALPS_MCANALYZE_EXPORT_HELPER( < int > , function_name_py, function_name_c )                                            \
    ALPS_MCANALYZE_EXPORT_HELPER( < std::vector < int > > , function_name_py, function_name_c )   


#define ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(function_name_py, function_name_c)                       \
    ALPS_MCANALYZE_EXPORT_SCALAR_AND_VECTOR( alps::alea::mcdata , function_name_py, function_name_c )                        \
    ALPS_MCANALYZE_EXPORT_SCALAR_AND_VECTOR( alps::alea::mctimeseries , function_name_py, function_name_c )             \
    ALPS_MCANALYZE_EXPORT_SCALAR_AND_VECTOR( alps::alea::mctimeseries_view , function_name_py, function_name_c )

#define ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_ONLY(function_name_py, function_name_c)                             \
    ALPS_MCANALYZE_EXPORT_SCALAR_ONLY( alps::alea::mcdata , function_name_py, function_name_c )                              \
    ALPS_MCANALYZE_EXPORT_SCALAR_ONLY( alps::alea::mctimeseries , function_name_py, function_name_c )                   \
    ALPS_MCANALYZE_EXPORT_SCALAR_ONLY( alps::alea::mctimeseries_view , function_name_py, function_name_c )

#define ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_VECTOR_ONLY(function_name_py, function_name_c)                             \
    ALPS_MCANALYZE_EXPORT_VECTOR_ONLY( alps::alea::mcdata , function_name_py, function_name_c )                              \
    ALPS_MCANALYZE_EXPORT_VECTOR_ONLY( alps::alea::mctimeseries , function_name_py, function_name_c )                   \
    ALPS_MCANALYZE_EXPORT_VECTOR_ONLY( alps::alea::mctimeseries_view , function_name_py, function_name_c )

#define ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_AND_VECTOR(function_name_py, function_name_c)                     \
    ALPS_MCANALYZE_EXPORT_SCALAR_AND_VECTOR( alps::alea::mctimeseries , function_name_py, function_name_c )             \
    ALPS_MCANALYZE_EXPORT_SCALAR_AND_VECTOR( alps::alea::mctimeseries_view , function_name_py, function_name_c )

#define ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_ONLY(function_name_py, function_name_c)                           \
    ALPS_MCANALYZE_EXPORT_SCALAR_ONLY( alps::alea::mctimeseries , function_name_py, function_name_c )                   \
    ALPS_MCANALYZE_EXPORT_SCALAR_ONLY( alps::alea::mctimeseries_view , function_name_py, function_name_c )
   

class_<alps::alea::value_with_error<double> >( "ValueWithError", init< optional<double, double> >() )                 
  .add_property("mean", &alps::alea::value_with_error<double>::mean)          
  .add_property("error", &alps::alea::value_with_error<double>::error)
  .def("__repr__", &alps::alea::print_value_with_error<double>)         
;


ALPS_MCANALYZE_EXPORT_MCTIMESERIES_CLASSES(double, MCScalarTimeseries)
ALPS_MCANALYZE_EXPORT_MCTIMESERIES_CLASSES(std::vector<double>, MCVectorTimeseries)

//ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_AND_VECTOR(cut_head, alps::alea::cut_head)
//ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_AND_VECTOR(cut_tail, alps::alea::cut_tail)

ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(size, alps::alea::size)

ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_ONLY(mean, alps::alea::mean)
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_VECTOR_ONLY(mean, alps::alea::mean_wrapper)

ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_ONLY(variance, alps::alea::variance)
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_VECTOR_ONLY(variance, alps::alea::variance_wrapper)
 
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(autocorrelation_distance, alps::alea::autocorrelation_distance)
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(autocorrelation_limit, alps::alea::autocorrelation_limit)
//ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_ONLY(exponential_autocorrelation_time_range, alps::alea::exponential_autocorrelation_time_range)
//ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_ONLY(exponential_autocorrelation_time_decay, alps::alea::exponential_autocorrelation_time_decay)
//ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_ONLY(integrated_autocorrelation_time_range, alps::alea::integrated_autocorrelation_time_range)
//ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_ONLY(integrated_autocorrelation_time_decay, alps::alea::integrated_autocorrelation_time_decay)
//ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(uncorrelated_error, alps::alea::error)
//ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(binning_error, alps::alea::binning_error)
//ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(running_mean, alps::alea::running_mean)
//ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(cutoff_mean, alps::alea::cutoff_mean)


#undef QUOTEME

#undef ALPS_MCANALYZE_EXPORT_HELPER
#undef ALPS_MCANALYZE_EXPORT_SCALAR_AND_VECTOR
#undef ALPS_MCANALYZE_EXPORT_SCALAR_ONLY
#undef ALPS_MCANALYZE_EXPORT_VECTOR_ONLY
#undef ALPS_MCANALYZE_EXPORT_VALUETYPE_FUNCTION
#undef ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR
#undef ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_ONLY
#undef ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_VECTOR_ONLY
#undef ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_AND_VECTOR
#undef ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_ONLY

#undef ALPS_MCANALYZE_EXPORT_MCTIMESERIES_CLASSES

}

