/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <boost/python/dict.hpp>

namespace alps { 
	namespace python {

		template<class T> T make_deepcopy(T const& x, boost::python::dict const& ) { return x; } 

	}
}