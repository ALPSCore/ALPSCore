/***************************************************************************
* ALPS++ library
*
* test/parameter/parameterlist.C   simple test program for ParameterList class
*
* $Id$
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

#include <alps/parameterlist.h>
#include <boost/throw_exception.hpp>
#include <iostream>
#include <stdexcept>

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

    alps::ParameterList params(std::cin);
    std::cout << params;

    {
      alps::OXDRFileDump od("parameterlist.dump");
      od << params;
    }

    params.clear();

    {
      alps::IXDRFileDump id("parameterlist.dump");
      id >> params;
    }

    std::cout << params;
    
#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& e)
{
  std::cerr << "Caught exception: " << e.what() << "\n";
  exit(-1);
}
catch (...)
{
  std::cerr << "Caught unknown exception\n";
  exit(-2);
}
#endif
}
