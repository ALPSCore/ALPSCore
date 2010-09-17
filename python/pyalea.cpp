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

typedef boost::variate_generator<boost::mt19937&, boost::uniform_01<double> > random_01;

using namespace boost::python;

namespace alps { 
  namespace alea {

    template <class T>
    static boost::python::str print_vector_list(std::vector<T> self)
    {
      boost::python::str s;
      for (typename std::vector<T>::iterator it = self.begin(); it != self.end(); ++it)
      {
        s += boost::python::str(*it);
        s += boost::python::str("\n");
      }
      return s;
    }


    // for interchanging purpose between numpy array and std::vector
    template <class T>  PyArray_TYPES getEnum();

    template <>   PyArray_TYPES getEnum<double>()              {  return PyArray_DOUBLE;      }
    template <>   PyArray_TYPES getEnum<long double>()         {  return PyArray_LONGDOUBLE;  }
    template <>   PyArray_TYPES getEnum<int>()                 {  return PyArray_INT;         }
    template <>   PyArray_TYPES getEnum<long>()                {  return PyArray_LONG;        }
    template <>   PyArray_TYPES getEnum<long long>()           {  return PyArray_LONG;        }
    template <>   PyArray_TYPES getEnum<unsigned long long>()  {  return PyArray_LONG;        }

    template <class T>
    boost::python::numeric::array convert2numpy_scalar(T value)
    {
        alps::python::numpy::import();                 // ### WARNING: forgetting this will end up in segmentation fault!
          
        npy_intp arr_size= 1;   // ### NOTE: npy_intp is nothing but just signed size_t
        boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(1, &arr_size, getEnum<T>())));  // ### NOTE: PyArray_SimpleNew is the new version of PyArray_FromDims
        void *arr_data= PyArray_DATA((PyArrayObject*) obj.ptr());
        memcpy(arr_data, &value, PyArray_ITEMSIZE((PyArrayObject*) obj.ptr()) * arr_size);
          
        return boost::python::extract<boost::python::numeric::array>(obj);
    }
      
    template <class T>
    boost::python::numeric::array convert2numpy_array(std::vector<T> vec)
    {
        alps::python::numpy::import();                 // ### WARNING: forgetting this will end up in segmentation fault!
          
        npy_intp arr_size= vec.size();   // ### NOTE: npy_intp is nothing but just signed size_t
        boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(1, &arr_size, getEnum<T>())));  // ### NOTE: PyArray_SimpleNew is the new version of PyArray_FromDims
        void *arr_data= PyArray_DATA((PyArrayObject*) obj.ptr());
        memcpy(arr_data, &vec.front(), PyArray_ITEMSIZE((PyArrayObject*) obj.ptr()) * arr_size);
          
        return boost::python::extract<boost::python::numeric::array>(obj);
    }
      
    template <class T>
    boost::python::numeric::array convertvalarray2numpy_array(std::valarray<T> vec)
    {
        alps::python::numpy::import();                 // ### WARNING: forgetting this will end up in segmentation fault!
          
        npy_intp arr_size= vec.size();   // ### NOTE: npy_intp is nothing but just signed size_t
        boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(1, &arr_size, getEnum<T>())));  // ### NOTE: PyArray_SimpleNew is the new version of PyArray_FromDims
        void *arr_data= PyArray_DATA((PyArrayObject*) obj.ptr());
        memcpy(arr_data, &vec[0], PyArray_ITEMSIZE((PyArrayObject*) obj.ptr()) * arr_size);
        
        return boost::python::extract<boost::python::numeric::array>(obj);
    }
      
    template <class T>
    std::vector<T> convert2vector(boost::python::object arr)
    {
      alps::python::numpy::import();                 // ### WARNING: forgetting this will end up in segmentation fault!

      std::size_t vec_size = PyArray_Size(arr.ptr());
      T * data = (T *) PyArray_DATA(arr.ptr());

      std::vector<T> vec(vec_size);
      memcpy(&vec.front(),data, PyArray_ITEMSIZE((PyArrayObject*) arr.ptr()) * vec_size);
      return vec;
    }
      
      template <typename T>
      std::valarray<T> convert2valarray(boost::python::object arr)
      {
          alps::python::numpy::import();                 // ### WARNING: forgetting this will end up in segmentation fault!
          
          std::size_t vec_size = PyArray_Size(arr.ptr());
          T * data = (T *) PyArray_DATA(arr.ptr());
          std::valarray<T> vec(vec_size);
          memcpy(&vec[0],data, PyArray_ITEMSIZE((PyArrayObject*) arr.ptr()) * vec_size);
          return vec;
      }

    template<typename T>
    class WrappedValarrayObservable
    {
        public:
        WrappedValarrayObservable(const std::string& name, int s=0)
        : obs(name,s)
        {}
        void operator<<(const boost::python::object& arr)
        {
            std::valarray< typename T:: value_type ::value_type > obj=convert2valarray<typename T:: value_type ::value_type >(arr);
            obs << obj;
        }
        std::string representation() const
        {
            return obs.representation();
        }
        
        boost::python::numeric::array mean() const 
        {
            std::valarray<typename T:: result_type ::value_type > mean = obs.mean();
            return convertvalarray2numpy_array<typename T:: result_type ::value_type >(mean);
        }
        
        boost::python::numeric::array error() const 
        {
            std::valarray<typename T:: result_type ::value_type > error = obs.error();
            return convertvalarray2numpy_array<typename T:: result_type ::value_type >(error);
        }
        boost::python::numeric::array tau() const 
        {
            std::valarray<typename T:: time_type ::value_type> tau = obs.tau();
            return convertvalarray2numpy_array<typename T:: result_type ::value_type >(tau);
        }
        boost::python::numeric::array variance() const 
        {
            std::valarray<typename T:: result_type ::value_type > variance = obs.variance();
            return convertvalarray2numpy_array<typename T:: result_type ::value_type >(variance);
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

BOOST_PYTHON_MODULE(pyalea_c) {
#define ALPS_PY_EXPORT_VECTOROBSERVABLE(class_name)                                                                             \
  class_<WrappedValarrayObservable< alps:: class_name > >(#class_name, init<std::string, optional<int> >())                     \
    .def("__repr__", &WrappedValarrayObservable< alps:: class_name >::representation)                                           \
    .def("__deepcopy__",  &alps::python::make_copy<WrappedValarrayObservable< alps::class_name > >)                             \
    .def("__lshift__", &WrappedValarrayObservable< alps::class_name >::operator<<)                                              \
    .def("save", &WrappedValarrayObservable< alps::class_name >::save)                                                          \
    .add_property("mean", &WrappedValarrayObservable< alps::class_name >::mean)                                                 \
    .add_property("error", &WrappedValarrayObservable< alps::class_name >::error)                                               \
    .add_property("tau", &WrappedValarrayObservable< alps::class_name >::tau)                                                   \
    .add_property("variance", &WrappedValarrayObservable< alps::class_name >::variance)                                         \
    .add_property("count", &WrappedValarrayObservable< alps::class_name >::count)                                               \
    ;
ALPS_PY_EXPORT_VECTOROBSERVABLE(IntVectorObservable)
ALPS_PY_EXPORT_VECTOROBSERVABLE(RealVectorObservable)
ALPS_PY_EXPORT_VECTOROBSERVABLE(IntVectorTimeSeriesObservable)
ALPS_PY_EXPORT_VECTOROBSERVABLE(RealVectorTimeSeriesObservable)
#undef ALPS_PY_EXPORT_VECTOROBSERVABLE
    
#define ALPS_PY_EXPORT_SIMPLEOBSERVABLE(class_name)                                                                                 \
  class_< alps:: class_name >(#class_name, init<std::string, optional<int> >())                                                     \
    .def("__deepcopy__",  &alps::python::make_copy<alps:: class_name >)                                                             \
    .def("__repr__", &alps:: class_name ::representation)                                                                           \
    .def("__lshift__", &alps:: class_name ::operator<<)                                                                             \
    .def("save", &alps::python::save_observable_to_hdf5<alps:: class_name >)                                                        \
    .add_property("mean", &alps:: class_name ::mean)                                                                                \
    .add_property("error", static_cast<alps:: class_name ::result_type(alps:: class_name ::*)() const>(&alps:: class_name ::error)) \
    .add_property("tau",&alps:: class_name ::tau)                                                                                   \
    .add_property("variance",&alps:: class_name ::variance)                                                                         \
    .add_property("count",&alps:: class_name ::count)                                                                               \
    ;                                                                                                                               \
       
ALPS_PY_EXPORT_SIMPLEOBSERVABLE(RealObservable)
ALPS_PY_EXPORT_SIMPLEOBSERVABLE(IntObservable)
ALPS_PY_EXPORT_SIMPLEOBSERVABLE(RealTimeSeriesObservable)
ALPS_PY_EXPORT_SIMPLEOBSERVABLE(IntTimeSeriesObservable)

#undef ALPS_PY_EXPORT_SIMPLEOBSERVABLE
    
    class_<boost::mt19937>("engine")
    .def("__deepcopy__",  &alps::python::make_copy<boost::mt19937>)
    .def("random", &boost::mt19937::operator())
    .def("max", &boost::mt19937::max )
    ;
    
    class_<boost::uniform_01<double> >("uniform")
    .def("__deepcopy__",  &alps::python::make_copy<boost::uniform_01<double> >)
    ;
    
    class_<random_01 >("random", init< boost::mt19937& , boost::uniform_01<double> >())
    .def("__deepcopy__",  &alps::python::make_copy<random_01 >)
    .def("random", static_cast<random_01::result_type(random_01::*)()>(&random_01::operator()))
    ;
    
   
  boost::python::def("convert2numpy_array_float",&convert2numpy_array<double>);
  boost::python::def("convert2numpy_array_int",&convert2numpy_array<int>);

  boost::python::def("convert2vector_double",&convert2vector<double>);
  boost::python::def("convert2vector_int",&convert2vector<int>);
}
