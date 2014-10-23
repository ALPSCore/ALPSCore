/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NUMERIC_INF_HEADER
#define ALPS_NUMERIC_INF_HEADER

#include <limits>

namespace alps {
    namespace numeric {

		template<typename T> struct inf {};

		template<> struct inf<double> {
    		operator double() const {
    			return std::numeric_limits<double>::infinity();
    		}
    	};

	}
}

#endif