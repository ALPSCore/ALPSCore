/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
