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

void write_set(const std::string& name, const alps::ModelLibrary& lib, 
               const alps::Parameters& p=alps::Parameters())
{
  alps::SiteBasisDescriptor<short> sitebasis=lib.site_basis(name);
  sitebasis.set_parameters(p);
  std::cout << "States of basis " << name << "=" << alps::SiteBasisStates<short>(sitebasis);
}

int main()
{

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif
    // create the library from an XML file
    alps::ModelLibrary lib(std::cin);

    // write all basis states
    write_set("fermion",lib);
    write_set("hardcore boson",lib);
    write_set("spin-1",lib);
    write_set("spin-1/2",lib);
    write_set("spin-3/2",lib);
    alps::Parameters p;
    p["S"]=2;
    write_set("spin",lib,p);

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
