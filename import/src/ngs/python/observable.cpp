/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#define PY_ARRAY_UNIQUE_SYMBOL pyngsobservable_PyArrayHandle

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/ngs/mcobservable.hpp>

#include <alps/ngs/boost_python.hpp>
#include <alps/ngs/detail/numpy_import.ipp>

#include <alps/alea/detailedbinning.h>

#include <boost/make_shared.hpp>

#include <valarray>

namespace alps {
    namespace detail {

        void observable_append(alps::mcobservable & self, boost::python::object const & data) {
            import_numpy();
            if (false);
            #define NGS_PYTHON_HDF5_CHECK_SCALAR(N)                                                                                                 \
                else if (std::string(data.ptr()->ob_type->tp_name) == N)                                                                            \
                    self << boost::python::extract< double >(data)();
            NGS_PYTHON_HDF5_CHECK_SCALAR("int")
            NGS_PYTHON_HDF5_CHECK_SCALAR("long")
            NGS_PYTHON_HDF5_CHECK_SCALAR("float")
            NGS_PYTHON_HDF5_CHECK_SCALAR("numpy.float64")
            else if (std::string(data.ptr()->ob_type->tp_name) == "numpy.ndarray" && PyArray_Check(data.ptr())) {
                PyArrayObject * ptr = (PyArrayObject *)data.ptr();
                if (!PyArray_ISNOTSWAPPED(ptr))
                    throw std::runtime_error("numpy array is not native" + ALPS_STACKTRACE);
                else if (!(ptr = PyArray_GETCONTIGUOUS(ptr)))
                    throw std::runtime_error("numpy array cannot be converted to continous array" + ALPS_STACKTRACE);
                self << std::valarray< double >(static_cast< double const *>(PyArray_DATA(ptr)), *PyArray_DIMS(ptr));
                Py_DECREF((PyObject *)ptr);
            } else
                throw std::runtime_error("unsupported type");
        }

        void observable_load(alps::mcobservable & self, alps::hdf5::archive & ar, std::string const & path) {
            std::string current = ar.get_context();
            ar.set_context(path);
            self.load(ar);
            ar.set_context(current);
        }

        alps::mcobservable create_RealObservable_export(std::string name) {
            return alps::mcobservable(boost::make_shared<alps::RealObservable>(name).get());
        }

        alps::mcobservable create_RealVectorObservable_export(std::string name) {
            return alps::mcobservable(boost::make_shared<alps::RealVectorObservable>(name).get());
        }
    }
}


BOOST_PYTHON_MODULE(pyngsobservable_c) {

    boost::python::def("createRealObservable", &alps::detail::create_RealObservable_export);
    boost::python::def("createRealVectorObservable", &alps::detail::create_RealVectorObservable_export);

    boost::python::class_<alps::mcobservable>(
        "observable",
        boost::python::no_init
    )
        .def("append", &alps::detail::observable_append)
        .def("merge", &alps::mcobservable::merge)
        .def("save", &alps::mcobservable::save)
        .def("load", &alps::detail::observable_load)
        .def("addToObservable", &alps::detail::observable_load)
    ;

}

