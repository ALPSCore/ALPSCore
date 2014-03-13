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

// fix for clang > 3.3
#include <alps/alea/simpleobseval.ipp>
#include <alps/alea/simpleobservable.ipp>
#include <alps/alea/abstractsimpleobservable.ipp>

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
          hdf5::archive ar(filename, "a");
          ar["/simulation/results/"+obs.representation()] << obs;
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
    ALPS_ALEA_FUNCTION_NUMPY_WRAPPER(uncorrelated_error)
    ALPS_ALEA_FUNCTION_NUMPY_WRAPPER(binning_error)

    #undef ALPS_ALEA_FUNCTION_NUMPY_WRAPPER

    template <class T>
    boost::python::str print_to_python (const T& IN) {
      std::ostringstream strs;
      strs << IN;
      return boost::python::str(strs.str());
    }

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

// mcdata docstrings
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

// mcanalyze docstrings
const char mctimeseries_docstring[] =
"This class is a simple class to store timeseries. It can be used with the free statistical functions in the pyalps.alea module.";

const char mctimeseries_view_docstring[] =
"This class is a view of a mctimeseries object. It does NOT copy the data so the object used to create this should not be deleted before the created object is deleted.";

const char mctimeseries_constructor_docstring[] =
"This constructor takes a MCTimeseries Object as argument and creates a reference to its data.";

const char mctimeseries_view_constructor_docstring[] =
"This constructor takes a MCTimeseriesView Object as argument and copies its reference to the data it is refering to.";

const char mcdata_constructor_docstring[] =
"This constructor takes a MCData Object as argument. It extracts the timeseries from the object.";

const char numpy_constructor_docstring[] =
"This constructor takes a numpy array as argument and constructs a timeseries from it.";

const char mctimeseries_timeseries_docstring[] =
"This returns the timeseries stored in the object as numpy array.";

const char size_docstring[] =
"This returns the size of the timeseries.";


const char std_pair_docstring[] =
"Export of a C++ std::pair<double, double>";

const char mcanalyze_mean_docstring[] =
"Takes any MCTimeseries or MCData object as argument. \n\
Returns the mean of the timeseries in a MCTimeseries object.";

const char mcanalyze_variance_docstring[] =
"Takes any MCTimeseries or MCData object as argument. \n\
Returns the variance of the timeseries in a MCTimeseries object.";

const char integrated_autocorrelation_time_docstring[] =
"Takes two arguments: A MCTimeseries object of the autocorrelation\nand a StdPairDouble object with a fit of the autocorrelation. \n\
Returns an estimate of the integrated autocorrelation time\nby summing up the autocorrelation as given and then integrating the tail using the fit.";

const char running_mean_docstring[] =
"Takes any MCTimeseries or MCData object as argument. \n\
Returns the running mean of the timeseries in a MCTimeseries object.";

const char reverse_running_mean_docstring[] =
"Takes any MCTimeseries or MCData object as argument. \n\
Returns the reverse running mean of the timeseries in a MCTimeseries object.";

// clang > 3.3 does not find the observable typedef, so instanciat it here ...  
template class alps::SimpleObservable<double, alps::DetailedBinning<double> >; // RealObservable
template class alps::SimpleObservable<double, alps::FixedBinning<double> >; // RealTimeSeriesObservable;
template class alps::SimpleObservable< std::valarray<double>, alps::DetailedBinning<std::valarray<double> > >; // RealVectorObservable;
template class alps::SimpleObservable< std::valarray<double>, alps::FixedBinning<std::valarray<double> > >; // RealVectorTimeSeriesObservable;
// instanciate the base classes of the observables
template class alps::AbstractSimpleObservable<double>;
template class alps::AbstractSimpleObservable<std::valarray<double> >;
// instanciate evaluators
template class alps::SimpleObservableEvaluator<double>;
template class alps::SimpleObservableEvaluator<std::valarray<double> >;

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
  class_<alps::alea::mctimeseries< type > >( QUOTEME(name) , mctimeseries_docstring)                                                                       \
    .def(init<boost::python::object>(numpy_constructor_docstring))                                                                                 \
    .def(init<alps::alea::mcdata< type > >(mcdata_constructor_docstring))                                                                                                 \
    .def("timeseries", &alps::alea::mctimeseries< type >::timeseries_python, mctimeseries_timeseries_docstring)   \
    .add_property("size", &alps::alea::mctimeseries< type >::size, size_docstring)           \
    .def("__repr__", &alps::alea::print_to_python<alps::alea::mctimeseries< type > >)  \
  ;                                                                                                                                     \
                                                                                                                                      \
  class_<alps::alea::mctimeseries_view< type > >( QUOTEME(name##View), mctimeseries_view_docstring, init< alps::alea::mctimeseries< type > >(mctimeseries_constructor_docstring))               \
    .def(init< alps::alea::mctimeseries_view< type > >(mctimeseries_view_constructor_docstring))                                                                        \
    .def("timeseries", &alps::alea::mctimeseries_view< type >::timeseries_python, numpy_constructor_docstring)                                                          \
    .add_property("size", &alps::alea::mctimeseries_view< type >::size, size_docstring)           \
    .def("__repr__", &alps::alea::print_to_python<alps::alea::mctimeseries_view< type > >)  \
  ; 


#define ALPS_MCANALYZE_EXPORT_HELPER(templateparms, function_name_py, function_name_c, docstring)                                       \
    def( QUOTEME ( function_name_py ), function_name_c templateparms , docstring);

#define ALPS_MCANALYZE_EXPORT_SCALAR_AND_VECTOR(container_type, function_name_py, function_name_c, docstring)                           \
    ALPS_MCANALYZE_EXPORT_HELPER( < container_type < double > > , function_name_py, function_name_c , docstring)                        \
    ALPS_MCANALYZE_EXPORT_HELPER( < container_type < std::vector < double > > > , function_name_py, function_name_c , docstring)                                        
/*    ALPS_MCANALYZE_EXPORT_HELPER( < container_type < int > > , function_name_py, function_name_c , docstring)                         \
    ALPS_MCANALYZE_EXPORT_HELPER( < container_type < std::vector < int > > > , function_name_py, function_name_c , docstring)*/

#define ALPS_MCANALYZE_EXPORT_SCALAR_ONLY(container_type, function_name_py, function_name_c, docstring)                                 \
    ALPS_MCANALYZE_EXPORT_HELPER( < container_type < double > > , function_name_py, function_name_c , docstring)  

#define ALPS_MCANALYZE_EXPORT_VECTOR_ONLY(container_type, function_name_py, function_name_c, docstring)                                 \
    ALPS_MCANALYZE_EXPORT_HELPER( < container_type < std::vector < double > > > , function_name_py, function_name_c , docstring)    

#define ALPS_MCANALYZE_EXPORT_VALUETYPE_FUNCTION(function_name_py, function_name_c, docstring)                                          \
    ALPS_MCANALYZE_EXPORT_HELPER( < double > , function_name_py, function_name_c , docstring)                                           \
    ALPS_MCANALYZE_EXPORT_HELPER( < std::vector < double > > , function_name_py, function_name_c , docstring)
/*    ALPS_MCANALYZE_EXPORT_HELPER( < int > , function_name_py, function_name_c , docstring)                                            \
    ALPS_MCANALYZE_EXPORT_HELPER( < std::vector < int > > , function_name_py, function_name_c , docstring)*/


#define ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(function_name_py, function_name_c, docstring)                       \
    ALPS_MCANALYZE_EXPORT_SCALAR_AND_VECTOR( alps::alea::mcdata , function_name_py, function_name_c , docstring)                        \
    ALPS_MCANALYZE_EXPORT_SCALAR_AND_VECTOR( alps::alea::mctimeseries , function_name_py, function_name_c , docstring)             \
    ALPS_MCANALYZE_EXPORT_SCALAR_AND_VECTOR( alps::alea::mctimeseries_view , function_name_py, function_name_c , docstring)

#define ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_ONLY(function_name_py, function_name_c, docstring)                             \
    ALPS_MCANALYZE_EXPORT_SCALAR_ONLY( alps::alea::mcdata , function_name_py, function_name_c , docstring)                              \
    ALPS_MCANALYZE_EXPORT_SCALAR_ONLY( alps::alea::mctimeseries , function_name_py, function_name_c , docstring)                   \
    ALPS_MCANALYZE_EXPORT_SCALAR_ONLY( alps::alea::mctimeseries_view , function_name_py, function_name_c , docstring)

#define ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_VECTOR_ONLY(function_name_py, function_name_c, docstring)                             \
    ALPS_MCANALYZE_EXPORT_VECTOR_ONLY( alps::alea::mcdata , function_name_py, function_name_c , docstring)                              \
    ALPS_MCANALYZE_EXPORT_VECTOR_ONLY( alps::alea::mctimeseries , function_name_py, function_name_c , docstring)                   \
    ALPS_MCANALYZE_EXPORT_VECTOR_ONLY( alps::alea::mctimeseries_view , function_name_py, function_name_c , docstring)

#define ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_AND_VECTOR(function_name_py, function_name_c, docstring)                     \
    ALPS_MCANALYZE_EXPORT_SCALAR_AND_VECTOR( alps::alea::mctimeseries , function_name_py, function_name_c , docstring)             \
    ALPS_MCANALYZE_EXPORT_SCALAR_AND_VECTOR( alps::alea::mctimeseries_view , function_name_py, function_name_c , docstring)

#define ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_ONLY(function_name_py, function_name_c, docstring)                           \
    ALPS_MCANALYZE_EXPORT_SCALAR_ONLY( alps::alea::mctimeseries , function_name_py, function_name_c , docstring)                   \
    ALPS_MCANALYZE_EXPORT_SCALAR_ONLY( alps::alea::mctimeseries_view , function_name_py, function_name_c , docstring)



docstring_options doc_options;  // complete docstring  

class_<alps::alea::value_with_error<double> >( "ValueWithError", init< optional<double, double> >() )                 
  .add_property("mean", &alps::alea::value_with_error<double>::mean)          
  .add_property("error", &alps::alea::value_with_error<double>::error)
  .def("__repr__", &alps::alea::print_value_with_error<double>)         
;

class_<std::pair< double, double > > ( "StdPairDouble", std_pair_docstring, init<double, double>() )
  .def_readwrite("first", &std::pair<double, double>::first)
  .def_readwrite("second", &std::pair<double, double>::second)
;


doc_options.disable_cpp_signatures(); // no cpp signatures

ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(size, alps::size, size_docstring)

  // need scalar and vector seperate so that scalar -> float, vector -> numpy
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_ONLY(mean, alps::alea::mean, mcanalyze_mean_docstring)
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_VECTOR_ONLY(mean, alps::alea::mean_wrapper, mcanalyze_mean_docstring)

ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_ONLY(variance, alps::alea::variance, mcanalyze_variance_docstring)
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_VECTOR_ONLY(variance, alps::alea::variance_wrapper, mcanalyze_variance_docstring)

ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_ONLY(integrated_autocorrelation_time, alps::alea::integrated_autocorrelation_time, integrated_autocorrelation_time_docstring)

ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_ONLY(running_mean, alps::alea::running_mean, running_mean_docstring)
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_ONLY(reverse_running_mean, alps::alea::reverse_running_mean, reverse_running_mean_docstring)

ALPS_MCANALYZE_EXPORT_MCTIMESERIES_CLASSES(double, MCScalarTimeseries)
ALPS_MCANALYZE_EXPORT_MCTIMESERIES_CLASSES(std::vector<double>, MCVectorTimeseries)
//ALPS_MCANALYZE_EXPORT_MCTIMESERIES_CLASSES(alps::alea::value_with_error<double>, MCScalarTimeseriesWithError)


doc_options.disable_all(); // no doc

 
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(autocorrelation_distance, alps::alea::autocorrelation_distance, "")
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(autocorrelation_limit, alps::alea::autocorrelation_limit, "")

ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_ONLY(exponential_autocorrelation_time_distance, alps::alea::exponential_autocorrelation_time_distance, "")
ALPS_MCANALYZE_EXPORT_MCTIMESERIES_FUNCTION_SCALAR_ONLY(exponential_autocorrelation_time_limit, alps::alea::exponential_autocorrelation_time_limit, "")

ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(cut_head_distance, alps::alea::cut_head_distance, "")
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_ONLY(cut_head_limit, alps::alea::cut_head_limit, "")

ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_AND_VECTOR(cut_tail_distance, alps::alea::cut_tail_distance, "")
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_ONLY(cut_tail_limit, alps::alea::cut_tail_limit, "")

ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_ONLY(uncorrelated_error, alps::alea::uncorrelated_error, "")
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_VECTOR_ONLY(uncorrelated_error, alps::alea::uncorrelated_error_wrapper, "")

ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_SCALAR_ONLY(binning_error, alps::alea::binning_error, "")
ALPS_MCANALYZE_EXPORT_TIMESERIES_FUNCTION_VECTOR_ONLY(binning_error, alps::alea::binning_error_wrapper, "")


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

