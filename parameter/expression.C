/***************************************************************************
* PALM++/palm library
*
* example/palm/expression.C   simple test program for expression class
*
* $Id$
*
* Copyright (C) 2001-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include <alps/expression.h>

#include <boost/throw_exception.hpp>
#include <iostream>
#include <stdexcept>

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif
  alps::Parameters parms;
  std::cin >> parms;
  std::cout << "Parameters:\n" << parms << std::endl;
  alps::check_character(std::cin,'%',"Expected a %-sign separating parameters from expressions");
  while (std::cin) {
    alps::Expression expr(std::cin);
    if (!expr.can_evaluate(parms))
      std::cout << "Cannot evaluate [" << expr << "]." << std::endl;
    else 
      std::cout << "The value of [" << expr << "] is " << expr.value(parms) 
		<< std::endl;
    char c;
    std::cin >> c;
    if (c!=',')
      break;
  }
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
