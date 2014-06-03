/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUMERIC_DOUBLE2INT_HPP
#define ALPS_NUMERIC_DOUBLE2INT_HPP

#include <boost/numeric/conversion/converter.hpp>

namespace alps { namespace numeric {

//
// double2int
//

/// \brief rounds a floating point value to the nearest integer
/// ex) double2int(3.6) -> 3
///     double2int(1.2) -> 1
///     duoble2int(-0.7) -> -1 (!= int(-0.7 + 0.5))
///
/// \return nearest integer of the input
inline int double2int(double in) {
  typedef boost::numeric::converter<int, double, boost::numeric::conversion_traits<int, double>,
    boost::numeric::def_overflow_handler, boost::numeric::RoundEven<double> > converter;
  return converter::convert(in);
}

} } // end namespace alps::numeric

#endif // ALPS_NUMERIC_DOUBLE2INT_HPP
