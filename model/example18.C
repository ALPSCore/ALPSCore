/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2005 by Matthias Troyer <troyer@itp.phys.ethz.ch>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/* $Id$ */

#include <alps/model.h>
#include <fstream>
#include <iostream>


void test(std::string const& name)
{
  // create the library from an XML file
  std::ifstream in("../../lib/xml/models.xml");
  alps::ModelLibrary lib(in);

  // get operators in one bond term 
  std::cout << "Model: " << name << "\n";
  std::cout << "Operator names:\n";
  std::set<std::string> names = lib.get_hamiltonian(name,alps::Parameters(),true).bond_term().operator_names();
  std::copy(names.begin(),names.end(),std::ostream_iterator<std::string>(std::cout,"\n"));
  
  std::cout << "\nSplit terms:\n\n";
  
  typedef std::vector<boost::tuple<alps::Term,alps::SiteOperator,alps::SiteOperator > > V;
  alps::SiteBasisDescriptor<short> b = lib.get_hamiltonian(name,alps::Parameters(),true).basis().site_basis();
  V  ops = lib.get_hamiltonian(name,alps::Parameters(),true).bond_term().split(b,b);
  for (V::const_iterator it=ops.begin(); it!=ops.end();++it)
    std::cout << "Prefactor: " << it->get<0>() << "\nSite 1: " << it->get<1>().term() << "\nSite 2: " << it->get<2>().term() << "\n\n";
}


int main()
{

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

  test("spin");
  test("spinless fermions");
  test("fermion Hubbard");

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
  return 0;
}
