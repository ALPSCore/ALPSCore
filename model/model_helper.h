/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

#ifndef ALPS_MODEL_MODEL_HELPER_H
#define ALPS_MODEL_MODEL_HELPER_H

#include <alps/model/modellibrary.h>

namespace alps {

template <class I=short>
class model_helper
{
public:  

  typedef BasisDescriptor<I> basis_descriptor_type;
  typedef half_integer<I> half_integer_type;
  typedef QuantumNumber<I> quantum_number_type;
  
  model_helper(alps::Parameters& p) // it updates the parameter object passed to it!
   : model_library_(p), 
     model_(model_library_.hamiltonian(p["MODEL"])) 
  {
    p.copy_undefined(model_.default_parameters());
    model_.set_parameters(p);
  }
  
  const ModelLibrary::OperatorDescriptorMap& simple_operators() const { 
    return model_library_.simple_operators();
  }
  
  HamiltonianDescriptor<I>& model() { return model_;}
  const HamiltonianDescriptor<I>& model() const { return model_;}
  basis_descriptor_type& basis() { return model().basis();}
  const basis_descriptor_type& basis() const { return model().basis();}
     
private:
   ModelLibrary model_library_;
   HamiltonianDescriptor<I> model_;
};

} // end namespace

#endif
