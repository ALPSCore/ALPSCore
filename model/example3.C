/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2004 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

int main()
{

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

    typedef alps::Expression Expression_;

    // create the library from an XML file
    std::ifstream in("../../lib/xml/models.xml");
    alps::ModelLibrary lib(in);
    alps::Parameters p;

     // write site term matrices
     std::cout << "HHardcoreBosonSite =\n"
               << alps::get_matrix(Expression_(),lib.get_hamiltonian("hardcore boson",p,true).site_term(),
                  lib.get_hamiltonian("hardcore boson").basis().site_basis()) << "\n";
    std::cout << "HSpinSite =\n" << alps::get_matrix(Expression_(),lib.get_hamiltonian("spin",p,true).site_term(),
                  lib.get_hamiltonian("spin").basis().site_basis()) << "\n";

    // write bond term matrices
    std::cout << "HHardcoreBosonBond =\n"
              << alps::get_matrix(Expression_(),lib.get_hamiltonian("hardcore boson",p,true).bond_term(),
                 lib.get_hamiltonian("hardcore boson").basis().site_basis(),
                 lib.get_hamiltonian("hardcore boson").basis().site_basis()) << "\n";
    std::cout << "HSpinBond =\n" << alps::get_matrix(Expression_(),lib.get_hamiltonian("spin",p,true).bond_term(),
                 lib.get_hamiltonian("spin").basis().site_basis(),lib.get_hamiltonian("spin").basis().site_basis()) << "\n";

     alps::Parameters parms;
     parms["Nmax"]=2; 
     alps::HamiltonianDescriptor<short> ham = lib.get_hamiltonian("boson Hubbard",parms,true);
     //ham.set_parameters(parms);
     std::cout << "HBosonSite =\n"
               << alps::get_matrix(Expression_(),ham.site_term(),ham.basis().site_basis()) << "\n";
     std::cout << "HBosonBond =\n"
               << alps::get_matrix(Expression_(),ham.bond_term(),
                  ham.basis().site_basis(),ham.basis().site_basis()) << "\n";

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
