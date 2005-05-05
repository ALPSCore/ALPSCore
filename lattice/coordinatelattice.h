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

#ifndef ALPS_LATTICE_COORDINATELATTICE_H
#define ALPS_LATTICE_COORDINATELATTICE_H

#include <alps/expression.h>
#include <alps/lattice/lattice.h>
#include <alps/lattice/simplelattice.h>
#include <alps/lattice/coordinate_traits.h>

#include <vector>

namespace alps {

template <class BASE = simple_lattice<>, class Vector = std::vector<double> >
class coordinate_lattice : public BASE
{
public:
  typedef BASE   parent_lattice_type;
  typedef typename lattice_traits<parent_lattice_type>::unit_cell_type
                 unit_cell_type;
  typedef typename lattice_traits<parent_lattice_type>::offset_type
                 offset_type;
  typedef typename lattice_traits<parent_lattice_type>::cell_descriptor
                 cell_descriptor;
  typedef Vector vector_type;
  typedef typename std::vector<vector_type>::const_iterator
                 basis_vector_iterator;
  
  coordinate_lattice() 
//    : basis_vectors_(1,Vector(1,boost::lexical_cast<typename Vector::value_type>(1))),
//      reciprocal_basis_vectors_(1,Vector(1,boost::lexical_cast<typename Vector::value_type>(1)))  
  {}
  
  template <class B2,class V2>
  coordinate_lattice(const coordinate_lattice<B2,V2>& l)
    : parent_lattice_type(l),
      basis_vectors_(alps::basis_vectors(l).second - 
                     alps::basis_vectors(l).first),
      reciprocal_basis_vectors_(alps::reciprocal_basis_vectors(l).second - 
                                alps::reciprocal_basis_vectors(l).first)
  {
    typename lattice_traits<coordinate_lattice<B2,V2> >::
      basis_vector_iterator it;
    int i = 0;
    for(it = alps::basis_vectors(l).first;
        it != alps::basis_vectors(l).second; ++it, ++i)
      std::copy(it->begin(), it->end(), std::back_inserter(basis_vectors_[i]));
    i = 0;
    for(it = alps::reciprocal_basis_vectors(l).first;
        it!=alps::reciprocal_basis_vectors(l).second; ++it, ++i)
      std::copy(it->begin(), it->end(),
                std::back_inserter(reciprocal_basis_vectors_[i]));
  }
  
  template <class InputIterator>
  coordinate_lattice(const unit_cell_type& u, InputIterator first,
                     InputIterator last)
    : parent_lattice_type(u), basis_vectors_(first, last) {}

  template <class InputIterator1, class InputIterator2>
  coordinate_lattice(const unit_cell_type& u, InputIterator1 first1,
                     InputIterator1 last1, InputIterator2 first2,
                     InputIterator2 last2)
    : parent_lattice_type(u), basis_vectors_(first1, last1),
      reciprocal_basis_vectors_(first2, last2) {}

  coordinate_lattice(const unit_cell_type& u) : parent_lattice_type(u) {}

  template <class B2, class V2>
  const coordinate_lattice& operator=(const coordinate_lattice<B2,V2>& l)
  {
    BASE::unit_cell() = l.unit_cell();
    basis_vectors_ = std::vector<vector_type>(
      alps::basis_vectors(l).first, alps::basis_vectors(l).second);
    reciprocal_basis_vectors_ = std::vector<vector_type>(
      alps::reciprocal_basis_vectors(l).first, alps::reciprocal_basis_vectors(l).second);
    return *this;
  }

  void set_parameters(const Parameters& p)
  {
    typename std::vector<vector_type>::iterator v_end = basis_vectors_.end();
    for (typename std::vector<vector_type>::iterator
           v = basis_vectors_.begin(); v != v_end; ++v) {
      typename vector_type::iterator b_end = v->end();
      for (typename vector_type::iterator b = v->begin(); b != b_end; ++b)
        *b = alps::evaluate<double>(*b, p);
    }
    v_end = reciprocal_basis_vectors_.end();
    for (typename std::vector<vector_type>::iterator
           v = reciprocal_basis_vectors_.begin(); v != v_end; ++v) {
      typename vector_type::iterator b_end = v->end();
      for (typename vector_type::iterator b = v->begin(); b != b_end; ++b)
        *b = alps::evaluate<double>(*b, p);
    }
  }

  void add_basis_vector(const vector_type& v) { basis_vectors_.push_back(v); }
  std::size_t num_basis_vectors() const { return basis_vectors_.size(); }
  std::pair<basis_vector_iterator, basis_vector_iterator>
  basis_vectors() const
  { return std::make_pair(basis_vectors_.begin(), basis_vectors_.end()); }

  void add_reciprocal_basis_vector(const vector_type& v)
  { reciprocal_basis_vectors_.push_back(v); }
  std::size_t num_reciprocal_basis_vectors() const
  { return reciprocal_basis_vectors_.size(); }
  std::pair<basis_vector_iterator, basis_vector_iterator>
  reciprocal_basis_vectors() const
  {
    return std::make_pair(reciprocal_basis_vectors_.begin(),
                          reciprocal_basis_vectors_.end());
  }

private:
  std::vector<vector_type> basis_vectors_;
  std::vector<vector_type> reciprocal_basis_vectors_;
};

template <class B, class V>
struct lattice_traits<coordinate_lattice<B,V> >
{
  typedef typename coordinate_lattice<B,V>::unit_cell_type  unit_cell_type;
  typedef typename coordinate_lattice<B,V>::cell_descriptor cell_descriptor;
  typedef typename coordinate_lattice<B,V>::offset_type     offset_type;
  typedef typename coordinate_lattice<B,V>::vector_type     vector_type;
  typedef typename coordinate_lattice<B,V>::basis_vector_iterator
    basis_vector_iterator;
};

} // end namespace alps

#endif // ALPS_LATTICE_COORDINATELATTICE_H
