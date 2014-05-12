/*****************************************************************************
*
* ALPS Project Applications
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@comp-phys.org>,
*                            Andreas Honecker <ahoneck@uni-goettingen.de>,
*                            Ryo IGARASHI <rigarash@hosi.phys.s.u-tokyo.ac.jp>
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

#ifndef ALPS_MODEL_HAMILTONIAN_MATRIX_HPP
#define ALPS_MODEL_HAMILTONIAN_MATRIX_HPP

#include <alps/model.h>
#include <alps/lattice.h>

#include <alps/utility/numeric_cast.hpp>
#include <alps/numeric/matrix_as_vector.hpp>
#include <alps/numeric/conj.hpp>
#include <alps/numeric/is_nonzero.hpp>
#include <alps/type_traits/is_symbolic.hpp>
#include <alps/multi_array.hpp>

#include <cmath>
#include <cstddef>

namespace alps {

namespace numeric {
    template <typename T, typename MemoryBlock>
    class matrix;
} // end namespace numeric

namespace detail {

    // The alps::numeric::matrix and ublas matrices behave differently on construction.
    // - ublas matrices are uninitalized -> call clear() afterwards
    // - alps::numeric::matrix is initialized after construction
    template <typename Matrix>
    Matrix initialized_matrix(Matrix const&, std::size_t const rows, std::size_t const cols)
    {
        Matrix m(rows,cols);
        m.clear();

        // Boost bug workaround
        // force at least one entry per row for sparse ublas matrices
        assert(rows == cols);
        for (std::size_t i=0;i<rows;++i) {
          m(i,i) +=1.;
          m(i,i) -=1.;
        }
        return m;
    }

    template <typename T, typename MemoryBlock>
    alps::numeric::matrix<T,MemoryBlock> initialized_matrix(alps::numeric::matrix<T,MemoryBlock> const&, std::size_t const rows, std::size_t const cols)
    {
        return alps::numeric::matrix<T,MemoryBlock>(rows,cols);
    }
} // end namespace detail

template <class M, class G = typename graph_helper<>::graph_type>
class hamiltonian_matrix
{
public:
  typedef M matrix_type;
  typedef typename matrix_type::value_type value_type;
  typedef typename graph_helper<>::graph_type graph_type;
  typedef typename graph_helper<>::site_descriptor site_descriptor;
  typedef typename graph_helper<>::bond_descriptor bond_descriptor;
  typedef typename graph_helper<>::site_iterator site_iterator;
  typedef typename graph_helper<>::bond_iterator bond_iterator;
  typedef typename graph_helper<>::vector_type vector_type;
  typedef typename model_helper<>::basis_descriptor_type basis_descriptor_type;
  typedef basis_states<short> basis_states_type;
  typedef bloch_basis_states<short> bloch_basis_states_type;
  typedef basis_states_type::value_type state_type;
  
  hamiltonian_matrix (Parameters const& parms);
  void set_parameters(Parameters const& p) { parms << p ; built_basis_=false; built_matrix_=false;}
  basis_states_type& states_vector() { if (!built_basis_) build_basis(); return states; }
  const basis_states_type& states_vector() const {if (!built_basis_) build_basis(); return states; }
  bloch_basis_states_type& bloch_states_vector() { if (!built_basis_) build_basis(); return bloch_states; }
  const bloch_basis_states_type& bloch_states_vector() const { if (!built_basis_) build_basis(); return bloch_states; }
  matrix_type& matrix() {if (!built_matrix_) build(); return matrix_;}
  const matrix_type& matrix() const {if (!built_matrix_) build(); return matrix_;}
  std::size_t dimension() const { if (!built_basis_) build_basis(); return uses_translation_invariance() ? bloch_states.size() : states.size();}
  void dostep();
  void print_basis(std::ostream& os) const
  {
    if (uses_translation_invariance())
      os << bloch_states_vector();
    else
      os << states_vector();
  }


  template <class STATES, class V, class W>
  void apply_operator(const STATES&, const SiteOperator& op, site_descriptor s, const V&, W&) const;

  template <class V, class W> 
  void apply_operator(const SiteOperator& op, site_descriptor s, const V& x, W& y) const
  {
    if (uses_translation_invariance())
      apply_operator(bloch_states,op,s,x,y);
    else
      apply_operator(states,op,s,x,y);
  }
  
  template <class V, class W> 
  void apply_operator(const BondOperator& op, bond_descriptor b, const V&, W&) const;

  template <class V, class W> 
  void apply_operator(const BondOperator& op, site_descriptor s1, site_descriptor s2, const V&, W&) const;

  template <class V, class W> 
  void apply_operator(const boost::multi_array<std::pair<value_type,bool>,4>& mat, site_descriptor s1, site_descriptor s2, const V& x, W& y) const
  {
    if (uses_translation_invariance())
      apply_operator(bloch_states,mat,s1,s2,x,y);
    else
      apply_operator(states,mat,s1,s2,x,y);
  }

  template <class STATES, class V, class W>
  void apply_operator(const STATES&, const boost::multi_array<std::pair<value_type,bool>,4>& mat, site_descriptor s1, site_descriptor s2, const V& x, W& y) const;

  template <class V, class W> 
  void apply_operator(const SiteOperator& op, const V&, W&) const;
  
  template <class V, class W> 
  void apply_operator(const BondOperator& op, const V&, W&) const;

  template <class V, class W> 
  void apply_operator(const GlobalOperator& op, const V&, W&) const;

  template <class MM, class OP> 
  MM operator_matrix(const OP& op) const 
  {
    MM m(detail::initialized_matrix(MM(),dimension(),dimension()));
    add_operator_matrix(m,op);
    return m;
  }

 
  template <class MM, class OP> 
  void add_operator_matrix(MM& m, const OP& op) const 
  {
    numeric::matrix_as_vector<MM> v(m);
    apply_operator(op,v,v);
  }

 
  template <class MM, class OP, class D> 
  MM operator_matrix(const OP& op, D d) const 
  {
    MM m(detail::initialized_matrix(MM(),dimension(),dimension()));
    add_operator_matrix(m,op,d);
    return m;
  }

  template <class MM, class OP, class D> 
  void add_operator_matrix(MM& m, const OP& op, D d) const 
  {
    numeric::matrix_as_vector<MM> v(m);
    apply_operator(op,d,v,v);
  }

  template <class MM, class OP> 
  MM operator_matrix(const OP& op, site_descriptor s1, site_descriptor s2) const 
  {
    MM m(detail::initialized_matrix(MM(),dimension(),dimension()));
    add_operator_matrix(m,op,s1,s2);
    return m;
  }

  template <class MM, class OP> 
  void add_operator_matrix(MM& m, const OP& op, site_descriptor s1, site_descriptor s2) const 
  {
    numeric::matrix_as_vector<MM> v(m);
    apply_operator(op,s1,s2,v,v);
  }

  multi_array<value_type,2> local_matrix(const SiteOperator& op, site_descriptor s) const;
  multi_array<std::pair<value_type,bool>,4> local_matrix(const BondOperator& op, const bond_descriptor& b) const;
  multi_array<std::pair<value_type,bool>,4> local_matrix(const BondOperator& op, const site_descriptor& s1, const site_descriptor& s2) const;

  template <class V, class W> 
  void apply_operator(const std::string& name, bond_descriptor b, const V& x, W& y) const
  { apply_operator(model_.get_bond_operator(name),b,x,y); }

  template <class V, class W>
  void apply_operator(const std::string& op, site_descriptor s, const V& x, W& y) const;

  template <class V, class W> 
  void apply_operator(const std::string& name, const V& x, W& y) const;

  bool uses_translation_invariance() const { return parms.defined("TOTAL_MOMENTUM") && !static_cast<std::string>(parms["TOTAL_MOMENTUM"]).empty();}

protected:
  void build() const;
  void build_basis() const;

  mutable basis_states_type states;
  mutable bloch_basis_states_type bloch_states;

private:
  Parameters parms;
  mutable bool built_matrix_;
  mutable bool built_basis_;
  mutable matrix_type matrix_;
  mutable basis_states_descriptor<short> basis_;
  graph_helper<G> graph_;
  model_helper<> model_;
  
};


template <class M, class G>
hamiltonian_matrix<M,G>::hamiltonian_matrix(Parameters const& p)
  : parms(p)
  ,  built_matrix_(false)
  ,  built_basis_(false)
  ,  graph_(p)
  ,  model_(graph_,p,is_symbolic<value_type>::type::value)
{}    


template <class M, class G>
multi_array<typename hamiltonian_matrix<M,G>::value_type,2> hamiltonian_matrix<M,G>::local_matrix(const SiteOperator& op, site_descriptor s) const
{
  Parameters p(parms);
  if (graph_.inhomogeneous_sites()) {
    throw_if_xyz_defined(parms,s); // check whether x, y, or z is set
    p << graph_.coordinate_as_parameter(s); // set x, y and z
  }
  return get_matrix(typename hamiltonian_matrix<M,G>::value_type(),op,model_.model().basis().site_basis(graph_.site_type(s)),p);
}

template <class M, class G>
multi_array<std::pair<typename hamiltonian_matrix<M,G>::value_type,bool>,4> hamiltonian_matrix<M,G>::local_matrix(const BondOperator& op, const bond_descriptor& b) const
{
  unsigned int stype1 = graph_.site_type(graph_.source(b));
  unsigned int stype2 = graph_.site_type(graph_.target(b));
  Parameters p(parms);
  if (graph_.inhomogeneous_bonds()) {
    throw_if_xyz_defined(parms,b); // check whether x, y, or z is set
    p << graph_.coordinate_as_parameter(b); // set x, y and z
  }
  return get_fermionic_matrix(typename hamiltonian_matrix<M,G>::value_type(),op,model_.model().basis().site_basis(stype1),
                                      model_.model().basis().site_basis(stype2),p);  
}

template <class M, class G>
multi_array<std::pair<typename hamiltonian_matrix<M,G>::value_type,bool>,4> hamiltonian_matrix<M,G>::local_matrix(const BondOperator& op, const site_descriptor& s1, const site_descriptor& s2) const
{
  unsigned int stype1 = graph_.site_type(s1);
  unsigned int stype2 = graph_.site_type(s2);
  return get_fermionic_matrix(typename hamiltonian_matrix<M,G>::value_type(),op,model_.model().basis().site_basis(stype1),
                                      model_.model().basis().site_basis(stype2),parms);  
}

template <class M, class G> template <class STATES, class V, class W>
void hamiltonian_matrix<M,G>::apply_operator(const STATES& states, const SiteOperator& op, site_descriptor s, const V& x, W& y) const
{
  multi_array<value_type,2> mat = local_matrix(op,s);
  for (unsigned int i=0;i<dimension();++i) {           // loop basis states
    state_type state=states[i];               // get source state
    int is=state[s];                          // get site basis index
    for (unsigned int js=0;js<basis_[s].size();++js) { // loop over target site states
      value_type val=mat[is][js];                      // get matrix element
      if (numeric::is_nonzero(val)) {   // if matrix element is nonzero
        state_type newstate=state;            // prepare target state
        newstate[s]=js;                       // build target state
        std::complex<double> phase;
        std::size_t j;       
        boost::tie(j,phase) = states.index_and_phase(newstate);       // lookup target state
        if (j<dimension()) {
          val *= numeric_cast<value_type>(phase * states.normalization(j)/states.normalization(i));
          y[i] += val*x[j];                   // set matrix element
          simplify(y[i]);               // optionally simplify a symbolic expression
       }
      }
    }
  }
}

template <class M, class G> template <class V, class W>
void hamiltonian_matrix<M,G>::apply_operator(const BondOperator& op, bond_descriptor b, const V& x, W& y) const
{
  apply_operator(local_matrix(op,b),graph_.source(b),graph_.target(b),x,y);
}  


template <class M, class G> template <class V, class W>
void hamiltonian_matrix<M,G>::apply_operator(const BondOperator& op, site_descriptor s1, site_descriptor s2, const V& x, W& y) const
{
  apply_operator(local_matrix(op,s1,s2),s1,s2,x,y);
}


template <class M, class G> template <class STATES, class V, class W>
void hamiltonian_matrix<M,G>::apply_operator(const STATES& states, const boost::multi_array<std::pair<value_type,bool>,4>& mat, site_descriptor s1, site_descriptor s2, const V& x, W& y) const
{
  for (std::size_t i=0;i<dimension();++i) {             // loop over source states
    state_type state=states[i];                   // get source state
    int is1=state[s1];                            // get source site states
    int is2=state[s2];
    for (unsigned int js1=0;js1<basis_[s1].size();++js1) { // loop over target site states
      for (unsigned int js2=0;js2<basis_[s2].size();++js2) {
        value_type val=mat[is1][is2][js1][js2].first;      // get matrix element
        if (numeric::is_nonzero(val)) {              // if nonzero matrix element
          state_type newstate=state;              // prepare target state
          newstate[s1]=js1;                       // build target state
          newstate[s2]=js2;
          std::complex<double> phase;
          std::size_t j;       
          boost::tie(j,phase) = states.index_and_phase(newstate);       // lookup target state
          if (j<dimension()) {
            if (mat[is1][is2][js1][js2].second) {
              // calculate fermionic sign
              bool f=(s2>=s1);
              int start = std::min(s1,s2);
              int end = std::max(s1,s2);

              for (int i=start;i<end;++i)
                if (is_fermionic(model_.model().basis().site_basis(graph_.site_type(i)),basis_[i][state[i]]))
                  f=!f;
              if (f)
                val=-val;
            }
            val *= numeric_cast<value_type>(phase * states.normalization(j)/states.normalization(i));
            y[i] += val*x[j];                   // set matrix element
            simplify(y[i]);               // optionally simplify a symbolic expression
          }
        }
      }
    }
  }
}



template <class M, class G> template <class V, class W>
void hamiltonian_matrix<M,G>::apply_operator(const std::string& op, site_descriptor s, const V& x, W& y) const
{
  if (model_.has_site_operator(op))
    apply_operator(model_.get_site_operator(op),s,x,y);
  else
    apply_operator(SiteOperator(op+"(i)","i"),s,x,y);
}


template <class M, class G> template <class V, class W>
void hamiltonian_matrix<M,G>::apply_operator(const std::string& name, const V& x, W& y) const
{ 
  if (model_.has_site_operator(name))
    apply_operator(model_.get_site_operator(name),x,y); 
  else if (model_.has_bond_operator(name))
    apply_operator(model_.get_bond_operator(name),x,y);
  else if (model_.has_global_operator(name))
    apply_operator(model_.get_global_operator(name),x,y);
  else // assume site operator
    apply_operator(SiteOperator(name+"(i)","i"),x,y);
}


template <class M, class G> template <class V, class W>
void hamiltonian_matrix<M,G>::apply_operator(const SiteOperator& op, const V& x, W& y) const
{
  for (site_iterator it=graph_.sites().first; it!=graph_.sites().second ; ++it) 
    apply_operator(op,*it,x,y);
}
  
template <class M, class G> template <class V, class W>
void hamiltonian_matrix<M,G>::apply_operator(const BondOperator& op, const V& x, W& y) const
{
  for (bond_iterator it=graph_.bonds().first; it!=graph_.bonds().second ; ++it) 
    apply_operator(op,*it,x,y);
}

template <class M, class G> template <class V, class W>
void hamiltonian_matrix<M,G>::apply_operator(const GlobalOperator& op, const V& x, W& y) const
{
  // loop over sites
  for (site_iterator it=graph_.sites().first; it!=graph_.sites().second ; ++it)
    apply_operator(op.site_term(graph_.site_type(*it)),*it,x,y);

  // loop over bonds
  for (bond_iterator it=graph_.bonds().first; it!=graph_.bonds().second ; ++it) 
      apply_operator(op.bond_term(graph_.bond_type(*it)),*it,x,y);
}




template <class M, class G>
void hamiltonian_matrix<M,G>::build_basis() const
{
  basis_descriptor_type b = model_.basis();
  b.set_parameters(parms);
  basis_ = basis_states_descriptor<short>(b,graph_.graph());
  if (uses_translation_invariance()) {
    std::vector<Expression> k;
    read_vector_resize(parms["TOTAL_MOMENTUM"],k);
    ParameterEvaluator eval(parms);
    vector_type total_momentum;
    for (unsigned i=0;i<k.size();++i)
      total_momentum.push_back(std::real(k[i].value(eval)));
    bloch_states = bloch_basis_states_type(basis_,graph_.translations(total_momentum));
  }
  else
    states = basis_states_type(basis_);
  built_basis_ = true;
}    

template <class M, class G>
void hamiltonian_matrix<M,G>::build() const
{
  // build matrix
  if (!built_basis_)
    build_basis();
  Disorder::seed(parms.value_or_default("DISORDER_SEED",0));
  matrix_ = detail::initialized_matrix(matrix_type(),dimension(),dimension());
  built_matrix_ = true;
  add_operator_matrix(matrix_,model_.model());

  //std::cerr << "Time to build matrix: " << t.elapsed() << "\n";
/*
  if (this->parms.value_or_default("CHECK_SYMMETRY",false)) {
    std::cerr << "Checking symmetry\n";
    for (unsigned i=0;i<dimension();++i)
      for (unsigned j=0;j<i;++j)
        if (std::abs(static_cast<value_type>(matrix_(i,j))-numeric::conj(static_cast<value_type>(matrix_(j,i)))) > 1e-10) {
          std::cerr << "Symmetry problem: " << i << " " << j << " " << static_cast<value_type>(matrix_(i,j)) << " " << numeric::conj(static_cast<value_type>(matrix_(j,i))) << "\n";
          std::cout << basis() << "\n";
          std::cout << states << "\n";
          std::cout << bloch_states << "\n";
          std::cout << matrix() << "\n";
          std::abort();
        }
    std::cerr << "Checked symmetry\n";
  }
*/
}

} // end name space alps

#endif // ALPS_MODEL_MATRIX_HPP

