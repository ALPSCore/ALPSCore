/***************************************************************************
* ALPS++/model library
*
* model/basisstates.h    basis states for full lattice
*
* $Id$
*
* Copyright (C) 2003-2003 by Matthias Troyer <troyer@comp-phys.org>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This software is part of the ALPS library, published under the 
* ALPS Library License; you can use, redistribute it and/or modify 
* it under the terms of the License, either version 1 or (at your option) 
* any later version.
*
* You should have received a copy of the ALPS Library License along with 
* the ALPS Library; see the file License.txt. If not, the license is also 
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
**************************************************************************/

#ifndef ALPS_MODEL_BASISSTATES_H
#define ALPS_MODEL_BASISSTATES_H

#include <alps/model/basis.h>
#include <boost/integer/static_log2.hpp>
#include <algorithm>
#include <vector>

namespace alps {

template <class I, class S=StateDescriptor<I> >
class BasisStatesDescriptor : public std::vector<SiteBasisStates<I,S> >
{
public:
  typedef SiteBasisStates<I,S> site_state_type;
  typedef std::vector<site_state_type> base_type;
  typedef typename base_type::const_iterator const_iterator;
  BasisStatesDescriptor() {}
  template <class G> BasisStatesDescriptor(const BasisDescriptor<I>& b, const G& graph);
  const BasisDescriptor<I>& basis() const { return basis_;}
  const SiteBasisDescriptor<I>& site_basis(int i) const { return site_basis_[i];}
  bool set_parameters(const Parameters& p) { return basis_.set_parameters(p);}
private:
  BasisDescriptor<I> basis_;
  std::vector<SiteBasisDescriptor<I> > site_basis_;
};

template <class I, int N=1> class IntegerState;

template <class I, int N>
class IntegerState {
public:
  static const int bits = boost::static_log2<N>::value+1;
  static const int mask = (1<<bits)-1;
  typedef I representation_type;
  
  class reference {
  public:
    reference(I& s, int i) : state_(s), shift_(i*bits) {}
    operator int() const { return (state_ >> shift_) & mask_;}
    template <class T>
    reference& operator=(T x)
    {
      state_ &= ~(mask<<shift_);
      state |= ((mask & x)<<shift_);
      return *this;
    }
  private:
    I& state_;
    std::size_t shift_;
  };
  
  IntegerState(representation_type x=0) : state_(x) {}
  
  template <class J>
  IntegerState(const std::vector<J>& x) : state_(0)
  { 
    for (int i=0;i<x.size();++i)  
      state_ |=(x[i]<<(i*bits));
  }
  int operator[](int i) const { return (state_>>i)&mask;}
  reference operator[](int i) { return reference(state_,i);}
  operator representation_type() const { return state_;}
  representation_type state() const { return state_;}
private:
  representation_type state_;
};

template <class I>
class IntegerState<I,1> {
public:
  typedef I representation_type;
  
  class reference {
  public:
    reference(I& s, int i) : state_(s), mask_(1<<i) {}
    operator int() const { return (state_&mask_ ? 1 : 0);}
    template <class T>
    reference& operator=(T x)
    {
      if (x)
        state_|=mask_;
      else
        state_&=~mask_;
      return *this;
    }
  private:
    I& state_;
    I mask_;
  };
  
  IntegerState(representation_type x=0) : state_(x) {}
  
  template <class J>
  IntegerState(const std::vector<J>& x) : state_(0)
  { 
    for (int i=0;i<x.size();++i)  
      if(x[i])
        state_ |=(1<<i);
  }
  int operator[](int i) const { return (state_>>i)&1;}
  reference operator[](int i) { return reference(state_,i);}
  operator representation_type() const { return state_;}
  representation_type state() const { return state_;}
private:
  representation_type state_;
};


template <class I, int N>
bool operator == (IntegerState<I,N> x, IntegerState<I,N> y)
{ return x.state() == y.state(); }

template <class I, int N>
bool operator < (IntegerState<I,N> x, IntegerState<I,N> y)
{ return x.state() < y.state(); }

template <class I, class S=std::vector<I>, class SS=StateDescriptor<I> >
class BasisStates : public std::vector<S>
{
public:
  typedef std::vector<S> base_type;
  typedef typename base_type::const_iterator const_iterator;
  typedef S value_type;
  typedef typename base_type::size_type size_type;
  typedef BasisStatesDescriptor<I,SS> basis_type;

  template <class J>
  BasisStates(const BasisStatesDescriptor<I,SS>& b, 
              const std::vector<std::pair<std::string,half_integer<J> > >& c)
    : basis_(b)
  { build(c);}

  BasisStates(const BasisStatesDescriptor<I,SS>& b)
    : basis_(b)
  { 
    build(b.basis().constraints());
  }
                    
                
  inline size_type index(const value_type& x) const
  {
    const_iterator it = std::lower_bound(begin(), end(), x);
    return (*it==x ? it-begin() : size());
  }

  bool check_sort() const;
  const basis_type& basis() const { return basis_;}
private:
  template <class J>
  bool satisfies_quantumnumbers(const std::vector<I>& idx, 
                                const std::pair<std::string,half_integer<J> >&);
  template <class J>
  void build(const std::vector<std::pair<std::string,half_integer<J> > >&);
                    

  BasisStatesDescriptor<I,SS> basis_;
};


template <class I=unsigned int, class J=short, class S=IntegerState<I>, class SS=StateDescriptor<J> >
class LookupBasisStates : public BasisStates<J,S,SS>
{
public:
  typedef BasisStates<I,S,SS> base_type;
  typedef typename base_type::const_iterator const_iterator;
  typedef S value_type;
  typedef typename base_type::size_type size_type;
  typedef typename base_type::basis_type basis_type;

  LookupBasisStates(const BasisStatesDescriptor<J,SS>& b) 
    : BasisStates<J,S,SS>(b), use_lookup_(false) 
  {
    if (b.size()<=24) {
      use_lookup_=true;
      lookup_.resize(1<<b.size(),size());
      for (int i=0;i<size();++i)
        lookup_[operator[](i)]=i;
    }
  }
  
  inline size_type index(const value_type& x) const
  {
    if (use_lookup_)
      return lookup_[x];
    else
      return BasisStates<J,S,SS>::index(x);
  }

private:
  bool use_lookup_;
  std::vector<I> lookup_;
};


// -------------------------- implementation -----------------------------------


template <class I, class S> template <class G>
BasisStatesDescriptor<I,S>::BasisStatesDescriptor(const BasisDescriptor<I>& b, const G& g)
 : basis_(b)
{
  // construct SiteBasisStates for each site
  typename property_map<site_type_t,G,int>::const_type site_type(get_or_default(site_type_t(),g,0));
  for (typename boost::graph_traits<G>::vertex_iterator it=sites(g).first;it!=sites(g).second ; ++it) {
    const SiteBasisDescriptor<I>& sb=basis_.site_basis(site_type[*it]);
    site_basis_.push_back(sb);
    push_back(site_state_type(sb));
  }
}


template <class I, class S, class SS>
bool BasisStates<I,S,SS>::check_sort() const
{
  for (int i=0;i<size()-1;++i)
    if (!((*this)[i]<(*this)[i+1]))
      return false;
  return true;
}

template <class I, class S, class SS> template <class J>
bool BasisStates<I,S,SS>::satisfies_quantumnumbers(const std::vector<I>& idx, const std::pair<std::string, half_integer<J> >& constraint )
{
  half_integer<J> val=0;
  for (int i=0;i<basis_.size();++i)
    val += get_quantumnumber(basis_[i][idx[i]],constraint.first,basis_.site_basis(i));
  return val==constraint.second;
}

template <class I, class S, class SS> template<class J>
void BasisStates<I,S,SS>::build(const std::vector<std::pair<std::string,half_integer<J> > >& constraints)
{
  if (basis_.empty())
    return;
  std::vector<I> idx(basis_.size(),0);
  unsigned int last=idx.size()-1;
  while (true) {
    unsigned int k=last;
    while (idx[k]>=basis_[k].size() && k) {
      idx[k]=0;
      if (k==0)
        break;
      --k;
      ++idx[k];
    }
    if (k==0 && idx[k]>=basis_[k].size())
      break;
      
    bool satisfies=true;
    for (int i=0;i<constraints.size();++i)
      satisfies = satisfies && satisfies_quantumnumbers(idx,constraints[i]);

    if (satisfies)
      push_back(idx);
    ++idx[last];
  }
  if (!check_sort()) {
    std::sort(begin(),end());
    if (!check_sort())
      boost::throw_exception(std::logic_error("Basis not sorted correctly"));
  }
}

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I, class S, class SS>
inline std::ostream& operator<<(std::ostream& out, const alps::BasisStates<I,S,SS>& q)
{
  out << "{\n";
  for (typename alps::BasisStates<I,S>::const_iterator it=q.begin();it!=q.end();++it) {
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
inline std::ostream& operator<<(std::ostream& out, const alps::LookupBasisStates<I,J,S,SS>& q)
{
  out << "{\n";
  for (typename alps::BasisStates<I,S>::const_iterator it=q.begin();it!=q.end();++it)
    out << "[ " << *it << " ]\n";
  out << "}\n";
  return out;	
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
