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

#ifndef ALPS_PYTHON_NUMPY_ARRAY
#define ALPS_PYTHON_NUMPY_ARRAY

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <vector>
#include <valarray>

namespace alps { 
  namespace python {
    namespace numpy {

      void import();

      void convert(boost::python::object const & source, std::vector<double> & target);

      boost::python::numeric::array convert(double source);

      boost::python::numeric::array convert(std::vector<double> const & source);

      boost::python::numeric::array convert(std::vector<std::vector<double> > const & source);

      boost::python::numeric::array convert(std::vector<std::vector<std::vector<double> > > const & source);

      
      // for interchanging purpose between numpy array and std::vector
      template <class T>  inline PyArray_TYPES getEnum();
      
      template <>   PyArray_TYPES inline getEnum<double>()              {  return PyArray_DOUBLE;      }
      template <>   PyArray_TYPES inline getEnum<long double>()         {  return PyArray_LONGDOUBLE;  }
      template <>   PyArray_TYPES inline getEnum<int>()                 {  return PyArray_INT;         }
      template <>   PyArray_TYPES inline getEnum<long>()                {  return PyArray_LONG;        }
      template <>   PyArray_TYPES inline getEnum<long long>()           {  return PyArray_LONG;        }
      template <>   PyArray_TYPES inline getEnum<unsigned long long>()  {  return PyArray_LONG;        }
      
      template <class T>
      boost::python::numeric::array convert2numpy(T value)
      {
        import();                 // ### WARNING: forgetting this will end up in segmentation fault!
        
        npy_intp arr_size= 1;   // ### NOTE: npy_intp is nothing but just signed size_t
        boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(1, &arr_size, getEnum<T>())));  // ### NOTE: PyArray_SimpleNew is the new version of PyArray_FromDims
        void *arr_data= PyArray_DATA((PyArrayObject*) obj.ptr());
        memcpy(arr_data, &value, PyArray_ITEMSIZE((PyArrayObject*) obj.ptr()) * arr_size);
        
        return boost::python::extract<boost::python::numeric::array>(obj);
      }
      
      template <class T>
      boost::python::numeric::array convert2numpy(std::vector<T> const& vec)
      {
        alps::python::numpy::import();                 // ### WARNING: forgetting this will end up in segmentation fault!
        
        npy_intp arr_size= vec.size();   // ### NOTE: npy_intp is nothing but just signed size_t
        boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(1, &arr_size, getEnum<T>())));  // ### NOTE: PyArray_SimpleNew is the new version of PyArray_FromDims
        void *arr_data= PyArray_DATA((PyArrayObject*) obj.ptr());
        memcpy(arr_data, &vec.front(), PyArray_ITEMSIZE((PyArrayObject*) obj.ptr()) * arr_size);
        
        return boost::python::extract<boost::python::numeric::array>(obj);
      }
      
      template <class T>
      boost::python::numeric::array convert2numpy(std::valarray<T> vec)
      {
        import();                 // ### WARNING: forgetting this will end up in segmentation fault!
        
        npy_intp arr_size= vec.size();   // ### NOTE: npy_intp is nothing but just signed size_t
        boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(1, &arr_size, getEnum<T>())));  // ### NOTE: PyArray_SimpleNew is the new version of PyArray_FromDims
        void *arr_data= PyArray_DATA((PyArrayObject*) obj.ptr());
        memcpy(arr_data, &vec[0], PyArray_ITEMSIZE((PyArrayObject*) obj.ptr()) * arr_size);
        
        return boost::python::extract<boost::python::numeric::array>(obj);
      }
      
      template <class T>
      std::vector<T> convert2vector(boost::python::object arr)
      {
        import();                 // ### WARNING: forgetting this will end up in segmentation fault!
        
        std::size_t vec_size = PyArray_Size(arr.ptr());
        T * data = (T *) PyArray_DATA(arr.ptr());
        
        std::vector<T> vec(vec_size);
        memcpy(&vec.front(),data, PyArray_ITEMSIZE((PyArrayObject*) arr.ptr()) * vec_size);
        return vec;
      }
      
      template <typename T>
      std::valarray<T> convert2valarray(boost::python::object arr)
      {
        import();                 // ### WARNING: forgetting this will end up in segmentation fault!
        
        std::size_t vec_size = PyArray_Size(arr.ptr());
        T * data = (T *) PyArray_DATA(arr.ptr());
        std::valarray<T> vec(vec_size);
        memcpy(&vec[0],data, PyArray_ITEMSIZE((PyArrayObject*) arr.ptr()) * vec_size);
        return vec;
      }
      
    }
  }
}

#endif
