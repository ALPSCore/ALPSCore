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

#include <alps/numeric/round.hpp>
#include <alps/model.h>
#include <alps/model/blochbasisstates.h>
#include <iostream>

int main()
{

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif
    alps::Parameters parms;
    std::cin >> parms;
    alps::ModelLibrary models(parms);
    alps::graph_helper<> lattices(parms);
    alps::HamiltonianDescriptor<short> ham(models.get_hamiltonian(parms["MODEL"]));
    parms.copy_undefined(ham.default_parameters());
    ham.set_parameters(parms);
    alps::basis_states_descriptor<short> basis(ham.basis(),lattices.graph());
    std::vector<alps::graph_helper<>::vector_type> k=lattices.translation_momenta();
    for (int ik=0;ik<k.size();++ik) {
      std::cout << "Momentum: ";
      for (int i=0;i<k[ik].size();++i)
        std::cout << k[ik][i] << " ";
      std::cout << "\n";
      std::vector<std::pair<std::complex<double>,std::vector<std::size_t> > > trans = lattices.translations(k[ik]);
      for (int i=0;i<trans.size();++i) {
        std::cout << "Translation " << i << " with phase "
                  << alps::numeric::round<1>(trans[i].first) << " maps ";
        for (int j=0;j<trans[i].second.size();++j)
          std::cout << j << "->" << trans[i].second[j] << " "; 
        std::cout << "\n";
      }
      alps::bloch_basis_states<short> states(basis,trans);
      std::cout << "Built states:\n" << states << std::endl;
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
  return 0;
}
