/***************************************************************************
* PALM++/xml library
*
* example/xml/parameters.C   test program for parameters XML handler
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
#include <iostream>
#include <cstdlib>

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif

  alps::ParameterList parameterlist;
  alps::ParameterListXMLHandler handler(parameterlist);
  
  alps::XMLParser parser(handler);
  parser.parse(std::cin);
  
  std::cout << "[Output to std::cout]\n";
  std::cout << parameterlist;
  
  std::cout << "\n[Output to alps::oxstream]\n";
  alps::oxstream oxs;
  oxs << parameterlist;

#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& exp) {
  std::cerr << exp.what() << std::endl;
  std::abort();
}
#endif
}
