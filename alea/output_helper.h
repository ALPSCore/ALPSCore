/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@itp.phys.ethz.ch>,
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

#ifndef ALPS_ALEA_OUTPUT_HELPER_H
#define ALPS_ALEA_OUTPUT_HELPER_H

#include <alps/config.h>
#include <alps/xml.h>
#include <boost/filesystem/path.hpp>
#include <boost/mpl/bool.hpp>
#include <iostream>

namespace alps {
template <typename FLAG> struct output_helper {};


template <>
struct output_helper<boost::mpl::true_>
{
  template <class X, class L> static void output(const X& b, std::ostream& out, const L&)
  {
    b.output_scalar(out);
  }

  template <class X> static void output(const X& b, std::ostream& out)
  {
    b.output_scalar(out);
  }

  template <class X> static void write_xml(const X& b, oxstream& oxs, const boost::filesystem::path& fn_hdf5)
  {
    b.write_xml_scalar(oxs, fn_hdf5);
  }
  template <class X, class IT> static void write_more_xml(const X& b, oxstream& oxs, IT)
  {
    b.write_scalar_xml(oxs);
  }
};

template <>
struct output_helper<boost::mpl::false_>
{
  template <class T, class L> static void output(const T& b, std::ostream& out, const L& label)
  {
    b.output_vector(out,label);
  }

  template <class T> static void output(const T& b, std::ostream& out)
  {
    b.output_vector(out);
  }

  template <class T> static void write_xml(const T& b, oxstream& oxs, const boost::filesystem::path& fn_hdf5)
  {
    b.write_xml_vector(oxs, fn_hdf5);
  }
  
  template <class X, class IT> static void write_more_xml(const X& b, oxstream& oxs, IT i)
  {
    b.write_vector_xml(oxs, i);
  }
};

} // end namespace alps


#endif // ALPS_ALEA_OUTPUT_HELPER_H
