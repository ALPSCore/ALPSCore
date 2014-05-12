/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2005 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_MODEL_BLOCHBASISSTATES_H
#define ALPS_MODEL_BLOCHBASISSTATES_H

#include <alps/model/basisstates.h>

namespace alps {

template <class I, class S=std::vector<I>, class SS=site_state<I> >
class bloch_basis_states : public std::vector<S>
{
  typedef std::vector<S> super_type;
public:
  typedef typename super_type::const_iterator const_iterator;
  typedef S value_type;
  typedef typename super_type::size_type size_type;
  typedef basis_states_descriptor<I,SS> basis_type;
  typedef std::vector<std::pair<std::complex<double>,std::vector<std::size_t> > > translation_type;

  bloch_basis_states() {}
  template <class J>
  bloch_basis_states(const basis_states_descriptor<I,SS>& b, const translation_type& t,
              const std::vector<std::pair<std::string,half_integer<J> > >& c)
    : basis_descriptor_(b)
  { build(t,c);}

  bloch_basis_states(const basis_states_descriptor<I,SS>& b, const translation_type& t)
    : basis_descriptor_(b)
  { 
    build(t,b.get_basis().constraints());
  }
                    
                
  inline std::pair<std::size_t,std::complex<double> > index_and_phase(const value_type& x) const
  {
    const_iterator it = std::lower_bound(full_list_.begin(), full_list_.end(), x);
    if (it==full_list_.end() || *it != x)
      return std::make_pair(size_type(super_type::size()),std::complex<double>(1.));
    size_type idx = it-full_list_.begin();
    return std::make_pair(representative_[idx],phase_[idx]);
  }

  const basis_type& basis() const { return basis_descriptor_;}
  
  double normalization(size_type i) const { return normalization_[i];}
  
  bool is_real() const
  {
    for (int i=0;i<phase_.size();++i)
      if (std::abs(std::imag(phase_[i]))>1.e-8)
        return false;
    return true;
  }
  std::vector<S> &full_list(){return full_list_;}
  
private:
  template <class J>
  bool satisfies_quantumnumbers(const std::vector<I>& idx, 
                                const std::pair<std::string,half_integer<J> >&);
  template <class J>
  void build(const translation_type&, const std::vector<std::pair<std::string,half_integer<J> > >&);
                    

  basis_states_descriptor<I,SS> basis_descriptor_;
  std::vector<S> full_list_;
  std::vector<std::size_t> representative_;
  std::vector<std::complex<double> > phase_;
  std::vector<double> normalization_;
};




// -------------------------- implementation -----------------------------------


template <class I, class S, class SS> template <class J>
bool bloch_basis_states<I,S,SS>::satisfies_quantumnumbers(const std::vector<I>& idx, const std::pair<std::string, half_integer<J> >& constraint )
{
  half_integer<J> val;
  for (int i=0;i<(int)basis_descriptor_.size();++i)
    val += get_quantumnumber(basis_descriptor_[i][idx[i]],constraint.first,basis_descriptor_.get_site_basis(i));
  return val==constraint.second;
}

template <class I, class S, class SS> template<class J>
void bloch_basis_states<I,S,SS>::build(const translation_type& trans, const std::vector<std::pair<std::string,half_integer<J> > >& constraints)
{    
  double l=std::sqrt(double(basis().size()));
  if (basis_descriptor_.empty())
    return;
  std::vector<I> idx(basis_descriptor_.size(),0);
  std::vector<int> fermionic(basis_descriptor_.size());
  unsigned int last=idx.size()-1;
    
  //
  // AML/AH: prepare a list of partial maxima and minima for each
  // constraint
  //
  multi_array<half_integer<J>,2> local_max(boost::extents[constraints.size()][idx.size()]);
  multi_array<half_integer<J>,2> local_min(boost::extents[constraints.size()][idx.size()]);
  
  multi_array<half_integer<J>,2> max_partial_qn_value(boost::extents[constraints.size()][idx.size()-1]);
  multi_array<half_integer<J>,2>    min_partial_qn_value(boost::extents[constraints.size()][idx.size()-1]);
  
  for (std::size_t ic=0;ic<constraints.size();++ic) {
      for(std::size_t is=0;is<idx.size();++is) {
          half_integer<J>& lmax=local_max[ic][is];
          half_integer<J>& lmin=local_min[ic][is];
          
          lmax=lmin=get_quantumnumber(basis_descriptor_[is][0],constraints[ic].first,basis_descriptor_.get_site_basis(is));
          
          for (std::size_t ib=1;ib<basis_descriptor_[is].size();++ib) {
              half_integer<J> val=get_quantumnumber(basis_descriptor_[is][ib],
                                                    constraints[ic].first,
                                                    basis_descriptor_.get_site_basis(is));
              if(lmax<val) lmax=val;
              if(lmin>val) lmin=val;
          }
      }
  }
  
  for (std::size_t ic=0;ic<constraints.size();++ic) {
      for(std::size_t ik=0;ik<last;++ik) {
          half_integer<J> max_val,min_val;
          for(std::size_t is=ik+1;is<idx.size();++is) {
              max_val+=local_max[ic][is];
              min_val+=local_min[ic][is];
          }
          max_partial_qn_value[ic][ik]=max_val;
          min_partial_qn_value[ic][ik]=min_val;
      }
  }  
  
  while (true) {
    unsigned int k=last;
    while (idx[k]>=(int)basis_descriptor_[k].size() && k) {
      idx[k]=0;
      if (k==0)
        break;
      --k;
      ++idx[k];
      
      //
      // AML/AH: truncates search tree if it is obvious that no new
      //         states can be found.
      // 
      bool breaked=false;
      if( idx[k]<(int)(basis_descriptor_[k].size() ) ){
          // let us see now if the new partial state
          // idx[0,k] is compatible with any of the partial
          // states idx[k+1,last]
          for (std::size_t ic=0;ic<constraints.size();++ic) {
              half_integer<J> val;
              for (std::size_t is=0;is<=k;++is)
                  val += get_quantumnumber(basis_descriptor_[is][idx[is]],
                                           constraints[ic].first,
                                           basis_descriptor_.get_site_basis(is));
              if (val+max_partial_qn_value[ic][k]<constraints[ic].second ||
                  val+min_partial_qn_value[ic][k]>constraints[ic].second) {
                  //impossible to satisfy constraint
                  breaked=true;
                  break;
              }
          }
          if(breaked)
              ++idx[k];
      }
      // end of new part
      
    }
    if (k==0 && idx[k]>=(int)basis_descriptor_[k].size())
      break;
      
    bool satisfies=true;
    for (int i=0;i<(int)constraints.size();++i)
      satisfies = satisfies && satisfies_quantumnumbers(idx,constraints[i]);
    
    if (satisfies) {        
      std::size_t representative_number = super_type::size(); 
      std::complex<double> ph=1.;
      std::complex<double> phase_sum=1.;
      int found_same=1;
      typename translation_type::const_iterator it=trans.begin();
      if (it!=trans.end())
        ++it;       // skip first trivial translation

      // search for translation giving back same state
      for (;it !=trans.end();++it) {
      // apply translation to state
        std::vector<I> translated_state(idx.size());
        for (int i=0;i<(int)it->second.size();++i)
          translated_state[it->second[i]]=idx[i];

        // if translated state is not smaller: next translation
        if (translated_state>idx)
          continue;
        // count fermion signs
        bool fermion_exchange=false;
        std::fill(fermionic.begin(),fermionic.end(),false);
        for (int i=basis().size()-1;i>=0;--i) {
          bool is=is_fermionic(basis()[i],idx[i]);
          if (is) {
            fermionic[it->second[i]]=is;
            if (std::accumulate(fermionic.begin(),fermionic.begin()+it->second[i],0) %2) 
              fermion_exchange=!fermion_exchange;
          }
        }
        
        double weight=(fermion_exchange ? -1. : 1.);

        // if is same after translation: increase count
        if (translated_state==idx) {
          ++found_same;
          phase_sum += weight*it->first;
          continue;
        }

        // look it up
        size_type i = index_and_phase(translated_state).first;
        if (i<super_type::size() && super_type::operator[](i)==translated_state) {
          // have found it and stored the phase
          representative_number = i;
          ph = weight*it->first;
          break; 
        }
      }
      
      bool storeit=true;
      if (representative_number == super_type::size()) { // not found
        double w = std::sqrt(std::abs(phase_sum));
        if (w<0.01)
          storeit=false;
        else if (std::abs(w-std::sqrt(double(found_same)))>0.1)
          std::cerr << "Strange weight: " << w << " " << found_same << "\n";
        if (storeit) {
          normalization_.push_back(w/l);
          std::vector<S>::push_back(idx);
        }
      }
      
      if (storeit) {
        representative_.push_back(representative_number);
        full_list_.push_back(idx);
        phase_.push_back(ph);
      }
    }

    ++idx[last];
  }

  if (super_type::size())
    for (int i=0;i<(int)super_type::size()-1;++i)
      if (!((*this)[i]<(*this)[i+1]))
        boost::throw_exception(std::logic_error("Bloch basis not sorted correctly"));
}

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I, class S, class SS>
inline std::ostream& operator<<(std::ostream& out, const alps::bloch_basis_states<I,S,SS>& q)
{
  out << "{\n";
  for (int i=0;i<(int)q.size();++i) {
    out << q.normalization(i) << "*[ ";
    unsigned int n=q[i].size();
    for (unsigned int j=0; j!=n;++j)
      out << q.basis()[j][q[i][j]] << " ";
    out << " ] + translations\n";
  }
  out << "}\n";
  return out;        
}


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
