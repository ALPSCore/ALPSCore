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

boost::multi_array<alps::Expression,2> bondmatrix(const alps::ModelLibrary lib, const std::string& name, 
                                                  const alps::Parameters& p=alps::Parameters())
{
  alps::HamiltonianDescriptor<short> ham=lib.hamiltonian(name);
  ham.set_parameters(p);
  int dim=ham.basis().site_basis().num_states();
  
  // get site and bond terms
  boost::multi_array<alps::Expression,2> sitematrix = 
    ham.site_term().matrix<alps::Expression>(ham.basis().site_basis(),lib.simple_operators());
  boost::multi_array<alps::Expression,4> bondtensor = 
    ham.bond_term().matrix<alps::Expression>(ham.basis().site_basis(), ham.basis().site_basis(),lib.simple_operators());
    
  // add site terms to bond terms
  for (int i=0;i<dim;++i)
    for (int j=0;j<dim;++j)
      for (int k=0;k<dim;++k) {
        bondtensor[i][j][i][k]+=sitematrix[j][k];
        bondtensor[j][i][k][i]+=sitematrix[j][k];
      }
      
  //convert tensor into matrix
  boost::multi_array<alps::Expression,2> bondmatrix(boost::extents[dim*dim][dim*dim]);
  for (int i=0;i<dim;++i)
    for (int j=0;j<dim;++j)
      for (int k=0;k<dim;++k)
        for (int l=0;l<dim;++l)
	  bondmatrix[i+j*dim][k+l*dim]=bondtensor[i][j][k][l];
  return bondmatrix;
}

int main()
{

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif
    // create the library from an XML file
    alps::ModelLibrary lib(std::cin);

    // calculate bond matrices 
    alps::Parameters parms;
    
    std::cout << "HHardcoreBoson = \n" << bondmatrix(lib,"hardcore boson") << "\n\n";
    parms["Nmax"]=2;
    std::cout << "HBoson = \n" << bondmatrix(lib,"boson",parms)  << "\n\n";
    parms["S"]="1/2";
    std::cout << "HSpinHalf = \n" << bondmatrix(lib,"spin")  << "\n\n";
    parms["S"]=1;
    std::cout << "HSpinOne = \n" << bondmatrix(lib,"spin",parms)  << "\n\n";
    

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
