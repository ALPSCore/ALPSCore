/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
