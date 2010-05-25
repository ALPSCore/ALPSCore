/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
*                            Matthias Troyer <troyer@itp.phys.ethz.ch>,
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
#define ALPS_HDF5_CLOSE_GREEDY

#include <alps/alea/detailedbinning.h>
#include <alps/alea/value_with_error.h>
#include <alps/alea/mcdata.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/python/make_copy.hpp>
#include <alps/python/save_observable_to_hdf5.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>

#include <boost/random.hpp>
#include<boost/random/uniform_01.hpp>

typedef boost::variate_generator<boost::mt19937&, boost::uniform_01<double> > random_01;

using namespace boost::python;


namespace alps { 
  namespace alea {

    // for printing support
    template <class T>
    inline static boost::python::str print_value_with_error(value_with_error<T> const & self)
    {
      return boost::python::str(boost::python::str(self.mean()) + " +/- " + boost::python::str(self.error()));
    }

    template <class T>
    static boost::python::str print_vector_with_error(value_with_error<std::vector<T> > self)
    {
      boost::python::str s;
      for (std::size_t index=0; index < self.size(); ++index)
      {
        s += boost::python::str(self.at(index));
        s += boost::python::str("\n");
      }
      return s;
    }

    template<class T>
    inline static boost::python::str print_vector_of_value_with_error(std::vector<value_with_error<T> > const & self)
    {
      typename std::vector<value_with_error<T> >::const_iterator it;

      boost::python::str s;
      for (it = self.begin(); it != self.end(); ++it)
      {
        s += print_value_with_error(*it);
        if (it != (self.end()-1))  {  s += '\n';  }
      }
      return s;
    }

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

    void import_numpy_array()               
    {  
      static bool inited = false;
      if (!inited) {
        import_array();  
        boost::python::numeric::array::set_module_and_type("numpy", "ndarray");  
        inited = true;
      }
    }
    
    
    template <class T>
    boost::python::numeric::array convert2numpy_scalar(T value)
    {
        import_numpy_array();                 // ### WARNING: forgetting this will end up in segmentation fault!
          
        npy_intp arr_size= 1;   // ### NOTE: npy_intp is nothing but just signed size_t
        boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(1, &arr_size, getEnum<T>())));  // ### NOTE: PyArray_SimpleNew is the new version of PyArray_FromDims
        void *arr_data= PyArray_DATA((PyArrayObject*) obj.ptr());
        memcpy(arr_data, &value, PyArray_ITEMSIZE((PyArrayObject*) obj.ptr()) * arr_size);
          
        return boost::python::extract<boost::python::numeric::array>(obj);
    }
      
    template <class T>
    boost::python::numeric::array convert2numpy_array(std::vector<T> vec)
    {
        import_numpy_array();                 // ### WARNING: forgetting this will end up in segmentation fault!
          
        npy_intp arr_size= vec.size();   // ### NOTE: npy_intp is nothing but just signed size_t
        boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(1, &arr_size, getEnum<T>())));  // ### NOTE: PyArray_SimpleNew is the new version of PyArray_FromDims
        void *arr_data= PyArray_DATA((PyArrayObject*) obj.ptr());
        memcpy(arr_data, &vec.front(), PyArray_ITEMSIZE((PyArrayObject*) obj.ptr()) * arr_size);
          
        return boost::python::extract<boost::python::numeric::array>(obj);
    }
      
    template <class T>
    boost::python::numeric::array convertvalarray2numpy_array(std::valarray<T> vec)
    {
        import_numpy_array();                 // ### WARNING: forgetting this will end up in segmentation fault!
          
        npy_intp arr_size= vec.size();   // ### NOTE: npy_intp is nothing but just signed size_t
        boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(1, &arr_size, getEnum<T>())));  // ### NOTE: PyArray_SimpleNew is the new version of PyArray_FromDims
        void *arr_data= PyArray_DATA((PyArrayObject*) obj.ptr());
        memcpy(arr_data, &vec[0], PyArray_ITEMSIZE((PyArrayObject*) obj.ptr()) * arr_size);
        
        return boost::python::extract<boost::python::numeric::array>(obj);
    }
      
    template <class T>
    std::vector<T> convert2vector(boost::python::object arr)
    {
      import_numpy_array();                 // ### WARNING: forgetting this will end up in segmentation fault!

      std::size_t vec_size = PyArray_Size(arr.ptr());
      T * data = (T *) PyArray_DATA(arr.ptr());

      std::vector<T> vec(vec_size);
      memcpy(&vec.front(),data, PyArray_ITEMSIZE((PyArrayObject*) arr.ptr()) * vec_size);
      return vec;
    }
      
      template <typename T>
      std::valarray<T> convert2valarray(boost::python::object arr)
      {
          import_numpy_array();                 // ### WARNING: forgetting this will end up in segmentation fault!
          
          std::size_t vec_size = PyArray_Size(arr.ptr());
          T * data = (T *) PyArray_DATA(arr.ptr());
          std::valarray<T> vec(vec_size);
          memcpy(&vec[0],data, PyArray_ITEMSIZE((PyArrayObject*) arr.ptr()) * vec_size);
          return vec;
      }

    // loading and extracting numpy arrays into vector_with_error
    #define IMPLEMENT_VECTOR_WITH_ERROR_CONSTRUCTION(TYPE) \
    template<> \
    value_with_error<std::vector<TYPE> >::value_with_error(boost::python::object const & mean_nparray, boost::python::object const & error_nparray) \
      : _mean(convert2vector<TYPE>(mean_nparray)) \
      , _error(convert2vector<TYPE>(error_nparray)) \
    {}

    IMPLEMENT_VECTOR_WITH_ERROR_CONSTRUCTION(int)
    IMPLEMENT_VECTOR_WITH_ERROR_CONSTRUCTION(long int)
    IMPLEMENT_VECTOR_WITH_ERROR_CONSTRUCTION(double)
    IMPLEMENT_VECTOR_WITH_ERROR_CONSTRUCTION(long double)

    #define IMPLEMENT_VECTOR_WITH_ERROR_GET(TYPE) \
    template<> \
    boost::python::object value_with_error<std::vector<TYPE> >::mean_nparray() const \
    { \
       return convert2numpy_array(_mean); \
    } \
    \
    template<> \
    boost::python::object value_with_error<std::vector<TYPE> >::error_nparray() const \
    { \
       return convert2numpy_array(_error); \
    } \

    IMPLEMENT_VECTOR_WITH_ERROR_GET(int)
    IMPLEMENT_VECTOR_WITH_ERROR_GET(long int)
    IMPLEMENT_VECTOR_WITH_ERROR_GET(double)
    IMPLEMENT_VECTOR_WITH_ERROR_GET(long double)
        
      
  /*   
#define IMPLEMENT_VECTOR_WITH_ERROR_CONSTRUCTION(TYPE) \
template<> \
value_with_error<std::valarray<TYPE> >::value_with_error(boost::python::object const & mean_nparray, std::valarray) \
: _mean(convert2vector<TYPE>(mean_nparray)) \
, _error(convert2vector<TYPE>(error_nparray)) \
{}
      
      IMPLEMENT_VECTOR_WITH_ERROR_CONSTRUCTION(int)
      IMPLEMENT_VECTOR_WITH_ERROR_CONSTRUCTION(long int)
      IMPLEMENT_VECTOR_WITH_ERROR_CONSTRUCTION(double)
      IMPLEMENT_VECTOR_WITH_ERROR_CONSTRUCTION(long double)    
   
*/      
    // for pickling support
    template<class T>
    struct value_with_error_pickle_suite : boost::python::pickle_suite
    {
      static boost::python::tuple getinitargs(const value_with_error<T>& v)
      {   
        return boost::python::make_tuple(v.mean(),v.error());
      }   
    };

    template<class T>
    struct vector_with_error_pickle_suite : boost::python::pickle_suite
    {
      static boost::python::tuple getinitargs(const value_with_error<std::vector<T> >& v)
      {
        return boost::python::make_tuple(v.mean_nparray(),v.error_nparray());
      }
    };

    template<class T>
    struct vector_of_value_with_error_pickle_suite : boost::python::pickle_suite
    {
      static boost::python::tuple getstate(const std::vector<value_with_error<T> > vec_of) 
      {
        value_with_error<std::vector<T> > vec_with = obtain_vector_with_error_from_vector_of_value_with_error<T>(vec_of);
        return boost::python::make_tuple(vec_with.mean_nparray(),vec_with.error_nparray());
      }

      static void setstate(std::vector<value_with_error<T> > & vec_of, boost::python::tuple state)
      {
        value_with_error<std::vector<T> > vec_with(state[0],state[1]);
        vec_of = obtain_vector_of_value_with_error_from_vector_with_error<T>(vec_with); 
      }
    };
      
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
    
        #define ALPS_PY_MCDATA_WRAPPER(member_name)                                                                                                        \
            template <class T> typename boost::enable_if<typename boost::is_scalar<T>::type, T>::type wrap_ ## member_name(mcdata<T> const & value) {      \
                return value. member_name ();                                                                                                              \
            }                                                                                                                                              \
            template <class T> boost::python::numeric::array wrap_ ## member_name(mcdata<std::vector<T> > const & value) {                                 \
                return convert2numpy_array(value. member_name ());                                                                                         \
            }
        ALPS_PY_MCDATA_WRAPPER(mean)
        ALPS_PY_MCDATA_WRAPPER(error)
        ALPS_PY_MCDATA_WRAPPER(tau)
        ALPS_PY_MCDATA_WRAPPER(variance)
        #undef ALPS_PY_MCDATA_WRAPPER
        template <typename T> boost::python::str print_mcdata(mcdata<T> const & self) {
            return boost::python::str(boost::python::str(self.mean()) + " +/- " + boost::python::str(self.error()));
        }
        template <typename T> boost::python::str print_mcdata(std::vector<mcdata<T> > const & self) {
            boost::python::str mean, error;
            for (typename std::vector<mcdata<T> >::const_iterator it = self.begin(); it != self.end(); ++it) {
                mean += (it != self.begin() ? ", " : "") + boost::python::str(it->mean());
                error += (it != self.begin() ? ", " : "") + boost::python::str(it->error());
            }
            return boost::python::str(boost::python::str(mean) + " +/- " + (error));
        }
        template <class T> std::vector<mcdata<T> > mcdata2vector_of_mcdata(mcdata<std::vector<T> > arg) {
            std::vector<mcdata<T> > res;
            for (std::size_t i = 0; i < arg.mean().size(); ++i)
                res.push_back(mcdata<T>(arg, i));
            return res;
        }
    }
}

using namespace alps::alea;
using namespace alps::numeric;

BOOST_PYTHON_MODULE(pyalea_c) {
    class_<mcdata<double> >("MCScalarData")
        .add_property("mean", static_cast<double(*)(mcdata<double> const &)>(&wrap_mean))
        .add_property("error", static_cast<double(*)(mcdata<double> const &)>(&wrap_error))
        .add_property("tau", static_cast<double(*)(mcdata<double> const &)>(&wrap_tau))
        .add_property("variance", static_cast<double(*)(mcdata<double> const &)>(&wrap_variance))
        .add_property("count", &mcdata<double>::count)
        .def("__repr__", static_cast<boost::python::str(*)(mcdata<double> const &)>(&print_mcdata))
        .def("__deepcopy__", &alps::python::make_copy<mcdata<double> >)
        .def("__abs__", static_cast<mcdata<double>(*)(mcdata<double>)>(&abs))
        .def("__pow__", static_cast<mcdata<double>(*)(mcdata<double>, mcdata<double>::element_type)>(&pow))
        .def(+self)
        .def(-self)
        .def(self += mcdata<double>())
        .def(self += double())
        .def(self -= mcdata<double>())
        .def(self -= double())
        .def(self *= mcdata<double>())
        .def(self *= double())
        .def(self /= mcdata<double>())
        .def(self /= double())
        .def(self + mcdata<double>())
        .def(mcdata<double>() + self)
        .def(self + double())
        .def(double() + self)
        .def(self - mcdata<double>())
        .def(mcdata<double>() - self)
        .def(self - double())
        .def(double() - self)
        .def(self * mcdata<double>())
        .def(mcdata<double>() * self)
        .def(self * double())
        .def(double() * self)
        .def(self / mcdata<double>())
        .def(mcdata<double>() / self)
        .def(self / double())
        .def(double() / self)
        .def("sq", static_cast<mcdata<double>(*)(mcdata<double>)>(&sq))
        .def("cb", static_cast<mcdata<double>(*)(mcdata<double>)>(&cb))
        .def("sqrt", static_cast<mcdata<double>(*)(mcdata<double>)>(&sqrt))
        .def("cbrt", static_cast<mcdata<double>(*)(mcdata<double>)>(&cbrt))
        .def("exp", static_cast<mcdata<double>(*)(mcdata<double>)>(&exp))
        .def("log", static_cast<mcdata<double>(*)(mcdata<double>)>(&log))
        .def("sin", static_cast<mcdata<double>(*)(mcdata<double>)>(&sin))
        .def("cos", static_cast<mcdata<double>(*)(mcdata<double>)>(&cos))
        .def("tan", static_cast<mcdata<double>(*)(mcdata<double>)>(&tan))
        .def("asin", static_cast<mcdata<double>(*)(mcdata<double>)>(&asin))
        .def("acos", static_cast<mcdata<double>(*)(mcdata<double>)>(&acos))
        .def("atan", static_cast<mcdata<double>(*)(mcdata<double>)>(&atan))
        .def("sinh", static_cast<mcdata<double>(*)(mcdata<double>)>(&sinh))
        .def("cosh", static_cast<mcdata<double>(*)(mcdata<double>)>(&cosh))
        .def("tanh", static_cast<mcdata<double>(*)(mcdata<double>)>(&tanh))
        .def("asinh", static_cast<mcdata<double>(*)(mcdata<double>)>(&asinh))
        .def("acosh", static_cast<mcdata<double>(*)(mcdata<double>)>(&acosh))
        .def("atanh", static_cast<mcdata<double>(*)(mcdata<double>)>(&atanh))
        .def("save", &mcdata<double>::save)
        .def("load", &mcdata<double>::load)
    ;
    class_<mcdata<std::vector<double> > >("MCVectorData")
        .add_property("mean", static_cast<boost::python::numeric::array(*)(mcdata<std::vector<double> > const &)>(&wrap_mean))
        .add_property("error", static_cast<boost::python::numeric::array(*)(mcdata<std::vector<double> > const &)>(&wrap_error))
        .add_property("tau", static_cast<boost::python::numeric::array(*)(mcdata<std::vector<double> > const &)>(&wrap_tau))
        .add_property("variance", static_cast<boost::python::numeric::array(*)(mcdata<std::vector<double> > const &)>(&wrap_variance))
        .add_property("count", &mcdata<std::vector<double> >::count)
        .def("__repr__", static_cast<boost::python::str(*)(mcdata<std::vector<double> > const &)>(&print_mcdata))
        .def("__deepcopy__", &alps::python::make_copy<mcdata<std::vector<double> > >)
        .def("__abs__", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&abs))
        .def("__pow__", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >, mcdata<double>::element_type)>(&pow))
        .def(+self)
        .def(-self)
        .def(self == mcdata<std::vector<double> >())
        .def(self += mcdata<std::vector<double> >())
        .def(self += std::vector<double>())
        .def(self -= mcdata<std::vector<double> >())
        .def(self -= std::vector<double>())
        .def(self *= mcdata<std::vector<double> >())
        .def(self *= std::vector<double>())
        .def(self /= mcdata<std::vector<double> >())
        .def(self /= std::vector<double>())
        .def(self + mcdata<std::vector<double> >())
        .def(mcdata<std::vector<double> >() + self)
        .def(self + std::vector<double>())
        .def(std::vector<double>() + self)
        .def(self - mcdata<std::vector<double> >())
        .def(mcdata<std::vector<double> >() - self)
        .def(self - std::vector<double>())
        .def(std::vector<double>() - self)
        .def(self * mcdata<std::vector<double> >())
        .def(mcdata<std::vector<double> >() * self)
        .def(self * std::vector<double>())
        .def(std::vector<double>() * self)
        .def(self / mcdata<std::vector<double> >())
        .def(mcdata<std::vector<double> >() / self)
        .def(self / std::vector<double>())
        .def(std::vector<double>() / self)
        .def(self + double())
        .def(double() + self)
        .def(self - double())
        .def(double() - self)
        .def(self * double())
        .def(double() * self)
        .def(self / double())
        .def(double() / self)
        .def("sq", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&sq))
        .def("cb", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&cb))
        .def("sqrt", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&sqrt))
        .def("cbrt", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&cbrt))
        .def("exp", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&exp))
        .def("log", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&log))
        .def("sin", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&sin))
        .def("cos", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&cos))
        .def("tan", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&tan))
        .def("asin", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&asin))
        .def("acos", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&acos))
        .def("atan", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&atan))
        .def("sinh", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&sinh))
        .def("cosh", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&cosh))
        .def("tanh", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&tanh))
        .def("asinh", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&asinh))
        .def("acosh", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&acosh))
        .def("atanh", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&atanh))
        .def("save", &mcdata<std::vector<double> >::save)
        .def("load", &mcdata<std::vector<double> >::load)
    ;
    class_<std::vector<mcdata<double> > >("VectorOfMCData")
        .def(vector_indexing_suite<std::vector<mcdata<double> > >())
        .def("__repr__", static_cast<boost::python::str(*)(std::vector<mcdata<double> > const &)>(&print_mcdata))
        .def("__deepcopy__", &alps::python::make_copy<std::vector<mcdata<double> > >)
        .def("__abs__", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&abs))
// TODO: fixit
//        .def("__pow__", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> > rhs, mcdata<double>::element_type)>(&pow))
        .def(self + std::vector<mcdata<double> >())
        .def(std::vector<mcdata<double> >() + self)
        .def(self + std::vector<double>())
        .def(std::vector<double>() + self)
        .def(self - std::vector<mcdata<double> >())
        .def(std::vector<mcdata<double> >() - self)
        .def(self - std::vector<double>())
        .def(std::vector<double>() - self)
        .def(self * std::vector<mcdata<double> >())
        .def(std::vector<mcdata<double> >() * self)
        .def(self * std::vector<double>())
        .def(std::vector<double>() * self)
        .def(self / std::vector<mcdata<double> >())
        .def(std::vector<mcdata<double> >() / self)
        .def(self / std::vector<double>())
        .def(std::vector<double>() / self)
        .def(self + double())
        .def(double() + self)
        .def(self - double())
        .def(double() - self)
        .def(self * double())
        .def(double() * self)
        .def(self / double())
        .def(double() / self)
        .def("sq", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&sq))
        .def("cb", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&cb))
        .def("sqrt", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&sqrt))
        .def("cbrt", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&cbrt))
        .def("exp", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&exp))
        .def("log", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&log))
        .def("sin", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&sin))
        .def("cos", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&cos))
        .def("tan", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&tan))
        .def("asin", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&asin))
        .def("acos", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&acos))
        .def("atan", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&atan))
        .def("sinh", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&sinh))
        .def("cosh", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&cosh))
        .def("tanh", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&tanh))
        .def("asinh", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&asinh))
        .def("acosh", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&acosh))
        .def("atanh", static_cast<std::vector<mcdata<double> >(*)(std::vector<mcdata<double> >)>(&atanh))
    ;
    boost::python::def("MCVectorData2VectorOfMCData", &mcdata2vector_of_mcdata<double>);

  class_<value_with_error<double> >("value_with_error",init<optional<double,double> >())
    .add_property("mean", &value_with_error<double>::mean)
    .add_property("error",&value_with_error<double>::error)  

    .def("__repr__", &print_value_with_error<double>)

    .def("__deepcopy__", &alps::python::make_copy<value_with_error<double> >)

    .def(+self)
    .def(-self)
    .def("__abs__", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&abs<double>))
    .def(self == value_with_error<double>())

    .def(self += value_with_error<double>())
    .def(self += double())
    .def(self -= value_with_error<double>())
    .def(self -= double())
    .def(self *= value_with_error<double>())
    .def(self *= double())
    .def(self /= value_with_error<double>())
    .def(self /= double())

    .def(self + value_with_error<double>())
    .def(self + double())
    .def(double() + self)
    .def(self - value_with_error<double>())
    .def(self - double())
    .def(double() - self)
    .def(self * value_with_error<double>())
    .def(self * double())
    .def(double() * self)
    .def(self / value_with_error<double>())
    .def(self / double())
    .def(double() / self)

    .def("__pow__", static_cast<value_with_error<double>(*)(value_with_error<double>, value_with_error<double>::element_type)>(&pow<double>))
    .def("sq", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&sq<double>))
    .def("cb", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&cb<double>))
    .def("sqrt", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&sqrt<double>))
    .def("cbrt", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&cbrt<double>))
    .def("exp", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&exp<double>))
    .def("log", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&log<double>))

    .def("sin", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&sin<double>))
    .def("cos", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&cos<double>))
    .def("tan", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&tan<double>))
    .def("asin", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&asin<double>))
    .def("acos", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&acos<double>))
    .def("atan", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&atan<double>))
    .def("sinh", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&sinh<double>))
    .def("cosh", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&cosh<double>))
    .def("tanh", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&tanh<double>))
    .def("asinh", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&asinh<double>))
    .def("acosh", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&acosh<double>))
    .def("atanh", static_cast<value_with_error<double>(*)(value_with_error<double>)>(&atanh<double>))

    .def_pickle(value_with_error_pickle_suite<double>())
    ;

  class_<value_with_error<std::vector<double> > >("vector_with_error",init<boost::python::object,boost::python::object>())
    .def(init<optional<std::vector<double>,std::vector<double> > >())

    .add_property("mean",&value_with_error<std::vector<double> >::mean_nparray)
    .add_property("error",&value_with_error<std::vector<double> >::error_nparray)

//    .add_property("mean",&mean_numpyarray<double>)
//    .add_property("error",&error_numpyarray<double>)

    .def("__repr__", &print_vector_with_error<double>)
    
    .def("__deepcopy__", &alps::python::make_copy<value_with_error<std::vector<double> > >)


    .def("__len__",&value_with_error<std::vector<double> >::size)         
    .def("append",&value_with_error<std::vector<double> >::push_back)     
    .def("push_back",&value_with_error<std::vector<double> >::push_back)  
    .def("insert",&value_with_error<std::vector<double> >::insert)    
    .def("pop_back",&value_with_error<std::vector<double> >::pop_back)   
    .def("erase",&value_with_error<std::vector<double> >::erase)        
    .def("clear",&value_with_error<std::vector<double> >::clear)         
    .def("at",&value_with_error<std::vector<double> >::at)

    .def(self + value_with_error<std::vector<double> >())
    .def(self + double())
    .def(double() + self)
    .def(self + std::vector<double>())
    .def(std::vector<double>() + self)
    .def(self - value_with_error<std::vector<double> >())
    .def(self - double())
    .def(double() - self)
    .def(self - std::vector<double>())
    .def(std::vector<double>() - self)
    .def(self * value_with_error<std::vector<double> >())
    .def(self * double())
    .def(double() * self)
    .def(self * std::vector<double>())
    .def(std::vector<double>() * self)
    .def(self / value_with_error<std::vector<double> >())
    .def(self / double())
    .def(double() / self)
    .def(self / std::vector<double>())
    .def(std::vector<double>() / self)

    .def(+self)
    .def(-self)
    .def("__abs__", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&abs<std::vector<double> >))
    .def("__pow__", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >, value_with_error<std::vector<double> >::element_type)>(&pow<std::vector<double> >))
    .def("sq", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&sq<std::vector<double> >))
    .def("cb", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&cb<std::vector<double> >))
    .def("sqrt", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&sqrt<std::vector<double> >))
    .def("cbrt", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&cbrt<std::vector<double> >))
    .def("exp", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&exp<std::vector<double> >))
    .def("log", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&log<std::vector<double> >))

    .def("sin", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&sin<std::vector<double> >))
    .def("cos", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&cos<std::vector<double> >))
    .def("tan", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&tan<std::vector<double> >))
    .def("asin", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&asin<std::vector<double> >))
    .def("acos", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&acos<std::vector<double> >))
    .def("atan", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&atan<std::vector<double> >))
    .def("sinh", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&sinh<std::vector<double> >))
    .def("cosh", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&cosh<std::vector<double> >))
    .def("tanh", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&tanh<std::vector<double> >))
    .def("asinh", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&asinh<std::vector<double> >))
    .def("acosh", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&acosh<std::vector<double> >))
    .def("atanh", static_cast<value_with_error<std::vector<double> >(*)(value_with_error<std::vector<double> >)>(&atanh<std::vector<double> >))

    .def_pickle(vector_with_error_pickle_suite<double>())

    ;


  class_<std::vector<value_with_error<double> > >("vector_of_value_with_error")
    .def(vector_indexing_suite<std::vector<value_with_error<double> > >())

    .def("__repr__", &print_vector_of_value_with_error<double>)

    .def("__deepcopy__", &alps::python::make_copy<std::vector<value_with_error<double> > >)

    .def(self + std::vector<value_with_error<double> >())
    .def(self + double())
    .def(double() + self)
    .def(self + std::vector<double>())
    .def(std::vector<double>() + self)
    .def(self - std::vector<value_with_error<double> >())
    .def(self - double())
    .def(double() - self)
    .def(self - std::vector<double>())
    .def(std::vector<double>() - self)
    .def(self * std::vector<value_with_error<double> >())
    .def(self * double())
    .def(double() * self)
    .def(self * std::vector<double>())
    .def(std::vector<double>() * self)
    .def(self / std::vector<value_with_error<double> >())
    .def(self / double())
    .def(double() / self)
    .def(self / std::vector<double>())
    .def(std::vector<double>() / self)

    .def(-self)

    .def("__abs__",&vec_abs<double>)

    .def("__pow__",&vec_pow<double>)
    .def("sq",&vec_sq<double>)
    .def("cb",&vec_cb<double>)
    .def("sqrt",&vec_sqrt<double>)
    .def("cbrt",&vec_cbrt<double>)
    .def("exp",&vec_exp<double>)
    .def("log",&vec_log<double>)

    .def("sin",&vec_sin<double>)
    .def("cos",&vec_cos<double>)
    .def("tan",&vec_tan<double>)
    .def("asin",&vec_asin<double>)
    .def("acos",&vec_acos<double>)
    .def("atan",&vec_atan<double>)
    .def("sinh",&vec_sinh<double>)
    .def("cosh",&vec_cosh<double>)
    .def("tanh",&vec_tanh<double>)
    .def("asinh",&vec_asinh<double>)
    .def("acosh",&vec_acosh<double>)
    .def("atanh",&vec_atanh<double>)

    .def_pickle(vector_of_value_with_error_pickle_suite<double>())

    ;

  boost::python::def("convert2vector_with_error",&obtain_vector_with_error_from_vector_of_value_with_error<double>);
  boost::python::def("convert2vector_of_value_with_error",&obtain_vector_of_value_with_error_from_vector_with_error<double>);

  class_<std::vector<double> >("vector")
    .def(vector_indexing_suite<std::vector<double> >())

    .def("__repr__", &print_vector_list<double>)
    ;
    
// TODO: use boost::lambda
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
