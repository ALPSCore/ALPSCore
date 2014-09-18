/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_ALEA_CONVERGENCE_H
#define ALPS_ALEA_CONVERGENCE_H

#include <alps/config.h>
#include <string>

namespace alps {

enum error_convergence {CONVERGED, MAYBE_CONVERGED, NOT_CONVERGED};

template<typename T> inline std::string convergence_to_text(T) {
	boost::throw_exception(std::logic_error("Not Implemented"));
	return std::string();
}

inline std::string convergence_to_text(int c)
{
  return (c==CONVERGED ? "yes" : c==MAYBE_CONVERGED ? "maybe" : c==NOT_CONVERGED ? "no" : "");
}

} // end namespace alps

#endif // ALPS_ALEA_CONVERGENCE_H
