/***************************************************************************
* PALM++/model library
*
* example/example1.C
*
* $Id$
*
* Copyright (C) 2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

#include <alps/model.h>
#include <iostream>

int main()
{

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif
    // create the library from an XML file
    alps::ModelLibrary lib(std::cin);

    // get operators in one bond term 
    std::set<alps::Term> ops = lib.hamiltonian("hardcore boson").bond_term().split(lib.simple_operators());
   
    std::copy(ops.begin(),ops.end(),std::ostream_iterator<alps::Term>(std::cout,"\n"));

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
