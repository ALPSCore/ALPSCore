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

     // write site term matrices
     std::cout << "HHardcoreBosonSite =\n" << lib.hamiltonian("hardcore boson").site_term().matrix<alps::Expression>(
              lib.hamiltonian("hardcore boson").basis().site_basis(),
 	     lib.simple_operators()) << "\n";
    std::cout << "HSpinSite =\n" << lib.hamiltonian("spin").site_term().matrix<alps::Expression>(
             lib.hamiltonian("spin").basis().site_basis(),
	     lib.simple_operators()) << "\n";
    
    // write bond term matrices
     std::cout << "HHardcoreBosonBond =\n" << lib.hamiltonian("hardcore boson").bond_term().matrix<alps::Expression>(
              lib.hamiltonian("hardcore boson").basis().site_basis(),
 	     lib.hamiltonian("hardcore boson").basis().site_basis(),
 	     lib.simple_operators()) << "\n";
    std::cout << "HSpinBond =\n" << lib.hamiltonian("spin").bond_term().matrix<alps::Expression>(
             lib.hamiltonian("spin").basis().site_basis(),
	     lib.hamiltonian("spin").basis().site_basis(),
	     lib.simple_operators()) << "\n";

     alps::Parameters parms;
     parms["Nmax"]=2;
     alps::HamiltonianDescriptor<short> ham = lib.hamiltonian("boson");
     ham.set_parameters(parms);
     std::cout << "HBosonSite =\n" << ham.site_term().matrix<alps::Expression>(
              ham.basis().site_basis(),lib.simple_operators()) << "\n";
     std::cout << "HBosonBond =\n" << ham.bond_term().matrix<alps::Expression>( ham.basis().site_basis(),
              ham.basis().site_basis(),lib.simple_operators()) << "\n";

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
