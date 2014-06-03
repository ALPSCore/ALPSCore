/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_PYTHON_MAKE_COPY_HPP
#define ALPS_PYTHON_MAKE_COPY_HPP

#include <boost/python/dict.hpp>
namespace alps { namespace python {

template<class T>
T make_copy(T const& x, boost::python::dict const& ) { return x; } 
 
} } // end namespace alps::python

#endif // ALPS_PYTHON_MAKE_COPY_HPP
