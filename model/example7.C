/***************************************************************************
* PALM++/model library
*
* example/example7.C
*
* $Id$
*
* Copyright (C) 2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>
*                       Axel Grzesik <axel@th.physik.uni-bonn.de>
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
#include <string>

void write_two_site_basis(const std::string& name, const alps::ModelLibrary& lib, 
			  const alps::Parameters& p=alps::Parameters())
{
  alps::SiteBasisDescriptor<short> sitebasis=lib.site_basis(name),
                                   sitebasis2=lib.site_basis(name);
  sitebasis.set_parameters(p);
  std::cout << "States of basis " << name << "=" << alps::SiteBasisStates<short>(sitebasis);
  for(int i=0;i<sitebasis.size();++i)
    sitebasis[i]+=sitebasis2[i];
  std::cout << "States of basis " << name << "=" << alps::SiteBasisStates<short>(sitebasis);
}

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif
  
  alps::ModelLibrary lib( std::cin );
  alps::Parameters p;
  p["NMax"]=2;
  write_two_site_basis("spin-1 boson",lib,p);

#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& exc) {
  std::cerr << exc.what() << "\n";
  //alps::comm_exit(true);
  return -1;
}
catch (...) {
  std::cerr << "Fatal Error: Unknown Exception!\n";
  return -2;
}
#endif
}
