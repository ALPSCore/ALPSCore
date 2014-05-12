/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2004 by Matthias Troyer <troyer@comp-phys.org>,
*                            Synge Todo <wistaria@comp-phys.org>
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

#ifndef ALPS_LATTICE_BOUNDARY_H
#define ALPS_LATTICE_BOUNDARY_H

#include <alps/config.h>
#include <alps/lattice/propertymap.h>
#include <alps/osiris/dump.h>
#include <boost/limits.hpp>
#include <boost/pending/property.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <string>
#include <vector>

namespace alps {

struct boundary_crossing {
  typedef unsigned int dimension_type;
  typedef int direction_type;

  boundary_crossing() : bc(0) {}
  operator bool() const { return bc!=0;}
  
  direction_type crosses(dimension_type d) const 
  { 
    return (bc&(1<<2*d)) ? +1 : ( (bc&(2<<2*d)) ? -1 : 0);
  }
  
  const boundary_crossing& set_crossing(dimension_type d, direction_type dir) 
  { 
    bc &= ~(3<<2*d);
    bc |= (dir>0 ? (1<<2*d) : (dir <0? (2<<2*d) : 0));
    return *this;
  }
  
  const  boundary_crossing& invert() 
  {
    integer_type rest=bc;
    int dim=0;
    while (rest) {
      invert(dim);
      dim++;
      rest >>=2;
    }
    return *this;
  }
  
  void save (ODump& dump) const { dump << bc;}
  void load (IDump& dump) { dump >> bc;}
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  { ar & bc; }
private:  
  typedef uint8_t integer_type;
  integer_type bc;
  const  boundary_crossing& invert(dimension_type d) {
    integer_type mask = 3<<2*d;
    if (bc&mask)
      bc^=mask;
    return *this;
  }
};

} // end namespace alps

#endif // ALPS_LATTICE_GRAPH_PROPERTIES_H
