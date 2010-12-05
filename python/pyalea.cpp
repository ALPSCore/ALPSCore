/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
*                            Lukas Gamper <gamperl@gmail.com>,
*                            Matthias Troyer <troyer@itp.phys.ethz.ch>
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

#include <boost/python.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <numpy/arrayobject.h>

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
          hdf5::oarchive ar(filename);
          ar << make_pvp("/simulation/results/"+obs.representation(), obs);
      }
      
      typename T::count_type count() const 
      {
          return obs.count();
      }
      
    private:
      T obs;
      
    };
  }
}

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
"the autocorrealtion time estimate of the recorded measurements.";

const char variance_docstring[] =
"the variance of all measurements recorded.";

const char count_docstring[] =
"the number of measurements recorded.";

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
    ;                                                                                                                               \
       
ALPS_PY_EXPORT_SIMPLEOBSERVABLE(RealObservable,observable_docstring,timeseries_constructor_docstring)
ALPS_PY_EXPORT_SIMPLEOBSERVABLE(RealTimeSeriesObservable,timeseries_observable_docstring,timeseries_constructor_docstring)

#undef ALPS_PY_EXPORT_SIMPLEOBSERVABLE

}
