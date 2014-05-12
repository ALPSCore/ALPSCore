/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_LATTICE_SIMPLECELL_H
#define ALPS_LATTICE_SIMPLECELL_H

#include <alps/config.h>
#include <alps/lattice/unitcell.h>
#include <alps/lattice/graph_traits.h>
#include <alps/lattice/dimensional_traits.h>
#include <alps/lattice/cell_traits.h>
#include <alps/utility/vectorio.hpp>

namespace alps {

template <class UnitCell=EmptyUnitCell, class Offset=typename std::vector<int> >
class simple_cell  {
public:
  typedef Offset offset_type;
  typedef UnitCell unit_cell_type;
  typedef typename alps::dimensional_traits<UnitCell>::dimension_type dimension_type;

  simple_cell() : dim_(0) {}
  simple_cell(const unit_cell_type& u, const offset_type& o)
   : dim_(alps::dimension(u)), offset_(o) {}
  
  const offset_type& offset() const { return offset_;}
  dimension_type dimension() { return dim_;}
private:
  dimension_type dim_;
  offset_type offset_;
};

template <class UnitCell,class Offset>
inline typename simple_cell<UnitCell,Offset>::dimension_type
dimension(const simple_cell<UnitCell,Offset>& c)
{
  return c.dimension();
}

} // end namespace alps

#endif // ALPS_LATTICE_SIMPLECELL_H
