/***************************************************************************
* PALM/xml library
*
* xml/simplexmlhandler.C   test program for SimpleXMLHandler class
*
* $Id$
*
* Copyright (C) 2001-2003 by Synge Todo <wistaria@comp-phys.org>,
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

#include <alps/parser/xmlhandler.h>
#include <alps/parser/xmlparser.h>
#include <cstdlib>
#include <iostream>
#include <string>

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif

  double v0;
  alps::SimpleXMLHandler<double> handler0("VALUE0", v0);

  double v1;
  alps::SimpleXMLHandler<double> handler1("VALUE1", v1, "value");

  alps::CompositeXMLHandler handler("TEST");
  handler.add_handler(handler0);
  handler.add_handler(handler1);

  alps::XMLParser parser(handler);
    
  parser.parse(std::cin);

  std::cout << v0 << std::endl
	    << v1 << std::endl;

#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& exp) {
  std::cerr << exp.what() << std::endl;
  std::abort();
}
#endif
}
