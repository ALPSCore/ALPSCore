/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2004 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_MODEL_BASISSTATES_H
#define ALPS_MODEL_BASISSTATES_H

#include <alps/model/basisdescriptor.h>
#include <alps/model/integer_state.h>
#include <algorithm>

namespace alps {

template <class I, class S=site_state<I> >
class basis_states_descriptor : public std::vector<site_basis<I,S> >
{
  typedef std::vector<site_basis<I,S> > super_type;
public:
  typedef site_basis<I,S> site_basis_type;
  typedef typename super_type::const_iterator const_iterator;
  basis_states_descriptor() {}
  template <class G> basis_states_descriptor(const BasisDescriptor<I>& b, const G& graph);
  const BasisDescriptor<I>& get_basis() const { return basis_descriptor_;}
  const SiteBasisDescriptor<I>& get_site_basis(int i) const { return site_basis_descriptor_[i];}
  bool set_parameters(const Parameters& p) { return basis_descriptor_.set_parameters(p);}
private:
  BasisDescriptor<I> basis_descriptor_;
  std::vector<SiteBasisDescriptor<I> > site_basis_descriptor_;
};


template <class I, class S=std::vector<I>, class SS=site_state<I> >
class basis_states : public std::vector<S>
{
  typedef std::vector<S> super_type;
public:
  typedef std::vector<S> super_type;
  typedef typename super_type::const_iterator const_iterator;
  typedef S value_type;
  typedef typename super_type::size_type size_type;
  typedef basis_states_descriptor<I,SS> basis_type;

  template <class J>
  basis_states(const basis_states_descriptor<I,SS>& b, 
              const std::vector<std::pair<std::string,half_integer<J> > >& c)
    : basis_descriptor_(b)
  { build(c);}

  basis_states(const basis_states_descriptor<I,SS>& b)
    : basis_descriptor_(b)
  { 
    build(b.get_basis().constraints());
  }
                    
                
  inline size_type index(const value_type& x) const
  {
    const_iterator it = std::lower_bound(super_type::begin(), super_type::end(), x);
    return (*it==x ? it-super_type::begin() : super_type::size());
  }

  bool check_sort() const;
  const basis_type& basis() const { return basis_descriptor_;}
private:
  template <class J>
  bool satisfies_quantumnumbers(const std::vector<I>& idx, 
                                const std::pair<std::string,half_integer<J> >&);
  template <class J>
  void build(const std::vector<std::pair<std::string,half_integer<J> > >&);
                    

  basis_states_descriptor<I,SS> basis_descriptor_;
};


template <class I=unsigned int, class J=short, class S=integer_state<I>, class SS=basis_states_descriptor<J> >
class lookup_basis_states : public basis_states<J,S,SS>
{
  typedef basis_states<I,S,SS> super_type;
public:
  typedef typename super_type::const_iterator const_iterator;
  typedef S value_type;
  typedef typename super_type::size_type size_type;
  typedef typename super_type::basis_type basis_type;

  lookup_basis_states(const basis_states_descriptor<J,SS>& b) 
    : basis_states<J,S,SS>(b), use_lookup_(false) 
  {
    if (b.size()<=24) {
      use_lookup_=true;
      lookup_.resize(1<<b.size(),super_type::size());
      for (int i=0;i<super_type::size();++i)
        lookup_[super_type::operator[](i)]=i;
    }
  }
  
  inline size_type index(const value_type& x) const
  {
    if (use_lookup_)
      return lookup_[x];
    else
      return basis_states<J,S,SS>::index(x);
  }

private:
  bool use_lookup_;
  std::vector<I> lookup_;
};


// -------------------------- implementation -----------------------------------


template <class I, class S> template <class G>
basis_states_descriptor<I,S>::basis_states_descriptor(const BasisDescriptor<I>& b, const G& g)
 : basis_descriptor_(b)
{
  // construct Sitebasis_states for each site
  typename property_map<site_type_t,G,int>::const_type site_type(get_or_default(site_type_t(),g,0));
  for (typename boost::graph_traits<G>::vertex_iterator it=sites(g).first;it!=sites(g).second ; ++it) {
    const SiteBasisDescriptor<I>& sb=basis_descriptor_.site_basis(site_type[*it]);
    site_basis_descriptor_.push_back(sb);
    push_back(site_basis_type(sb));
  }
}


template <class I, class S, class SS>
bool basis_states<I,S,SS>::check_sort() const
{
  for (int i=0;i<super_type::size()-1;++i)
    if (!((*this)[i]<(*this)[i+1]))
      return false;
  return true;
}

template <class I, class S, class SS> template <class J>
bool basis_states<I,S,SS>::satisfies_quantumnumbers(const std::vector<I>& idx, const std::pair<std::string, half_integer<J> >& constraint )
{
  half_integer<J> val=0;
  for (int i=0;i<basis_descriptor_.size();++i)
    val += get_quantumnumber(basis_descriptor_[i][idx[i]],constraint.first,basis_descriptor_.get_site_basis(i));
  return val==constraint.second;
}

template <class I, class S, class SS> template<class J>
void basis_states<I,S,SS>::build(const std::vector<std::pair<std::string,half_integer<J> > >& constraints)
{
  if (basis_descriptor_.empty())
    return;
  std::vector<I> idx(basis_descriptor_.size(),0);
  unsigned int last=idx.size()-1;
  while (true) {
    unsigned int k=last;
    while (idx[k]>=basis_descriptor_[k].size() && k) {
      idx[k]=0;
      if (k==0)
        break;
      --k;
      ++idx[k];
    }
    if (k==0 && idx[k]>=basis_descriptor_[k].size())
      break;
      
    bool satisfies=true;
    for (int i=0;i<constraints.size();++i)
      satisfies = satisfies && satisfies_quantumnumbers(idx,constraints[i]);

    if (satisfies)
      push_back(idx);
    ++idx[last];
  }
  if (!check_sort()) {
    std::sort(super_type::begin(),super_type::end());
    if (!check_sort())
      boost::throw_exception(std::logic_error("Basis not sorted correctly"));
  }
}

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I, class S, class SS>
inline std::ostream& operator<<(std::ostream& out, const alps::basis_states<I,S,SS>& q)
{
  out << "{\n";
  for (typename alps::basis_states<I,S>::const_iterator it=q.begin();it!=q.end();++it) {
    out << "[ ";
    //out << *it;
    unsigned int n=it->size();
    for (unsigned int i=0; i!=n;++i)
      out << q.basis()[i][(*it)[i]] << " ";
//      out << (*it)[i] << " ";
    out << " ]\n";
  }
  out << "}\n";
  return out;        
}

template <class I, class J, class S, class SS>
inline std::ostream& operator<<(std::ostream& out, const alps::lookup_basis_states<I,J,S,SS>& q)
{
  out << "{\n";
  for (typename alps::basis_states<I,S>::const_iterator it=q.begin();it!=q.end();++it)
    out << "[ " << *it << " ]\n";
  out << "}\n";
  return out;        
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
