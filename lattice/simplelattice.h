/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2005 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_LATTICE_SIMPLELATTICE_H
#define ALPS_LATTICE_SIMPLELATTICE_H

#include <alps/config.h>
#include <alps/lattice/cell_traits.h>
#include <alps/lattice/dimensional_traits.h>
#include <alps/lattice/coordinate_traits.h>
#include <alps/lattice/simplecell.h>
#include <alps/lattice/lattice.h>

namespace alps {

template<class UnitCell = EmptyUnitCell, class Cell = simple_cell<UnitCell> >
class simple_lattice
{
public:
  typedef UnitCell unit_cell_type;
  typedef Cell     cell_descriptor;
  typedef typename alps::dimensional_traits<unit_cell_type>::dimension_type
                   dimension_type;
  typedef typename alps::cell_traits<cell_descriptor>::offset_type
                   offset_type;
  
  simple_lattice() {}
  template <class U2, class C2>
  simple_lattice(const simple_lattice<U2,C2>& l)
    : unit_cell_(l.unit_cell()) {}
  simple_lattice(const unit_cell_type& c) : unit_cell_(c) {}
  
  template <class U2, class C2>
  const simple_lattice& operator=(const simple_lattice<U2,C2>& l)
  {
    unit_cell_ = l.unit_cell();
    return *this;
  }

  unit_cell_type& unit_cell() { return unit_cell_; }
  const unit_cell_type& unit_cell() const { return unit_cell_; }
  
  cell_descriptor cell(offset_type o) const 
  { return cell_descriptor(unit_cell_, o); }
  
  dimension_type dimension() const { return alps::dimension(unit_cell_); }

protected:
  unit_cell_type unit_cell_;
};

template <class U, class C>
struct lattice_traits<simple_lattice<U,C> >
{
  typedef typename simple_lattice<U,C>::unit_cell_type  unit_cell_type;
  typedef typename simple_lattice<U,C>::cell_descriptor cell_descriptor;
  typedef typename simple_lattice<U,C>::offset_type     offset_type;
};

template <class U, class C>
inline typename dimensional_traits<simple_lattice<U,C> >::dimension_type
dimension (const simple_lattice<U,C>& l)
{ return l.dimension(); }

template<class UnitCell, class Cell>
typename simple_lattice<UnitCell, Cell>::unit_cell_type&
unit_cell(simple_lattice<UnitCell, Cell>& l) { return l.unit_cell(); }

template<class UnitCell, class Cell>
const typename simple_lattice<UnitCell, Cell>::unit_cell_type&
unit_cell(const simple_lattice<UnitCell, Cell>& l) { return l.unit_cell(); }

} // end namespace alps

#endif // ALPS_LATTICE_SIMPLELATTICE_H
