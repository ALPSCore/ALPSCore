/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1999-2010 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>
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
