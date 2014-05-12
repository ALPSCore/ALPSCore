/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2009 by Matthias Troyer <troyer@comp-phys.org>,
*                            Axel Grzesik <axel@th.physik.uni-bonn.de>
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

#ifndef ALPS_MODEL_SITEBASISSTATES_H
#define ALPS_MODEL_SITEBASISSTATES_H

#include <alps/model/quantumnumber.h>
#include <alps/model/sitestate.h>
#include<boost/config.hpp>
#include <vector>
#include <iostream>

namespace alps {

template <class I, class STATE=site_state<I> >
class site_basis : public std::vector<STATE>
{
  typedef std::vector<STATE> super_type;
public:
  typedef STATE state_type;
  typedef std::vector<state_type> base_type;
  typedef typename base_type::const_iterator const_iterator;
  typedef typename base_type::value_type value_type;
  typedef typename base_type::size_type size_type;
  site_basis(const SiteBasisDescriptor<I>& b);
    
  size_type index(const value_type& x) const;
  const SiteBasisDescriptor<I>& basis() const { return basis_;}
  bool check_sort() const;

private:
  SiteBasisDescriptor<I> basis_;
};


template <class I, class STATE>
bool is_fermionic(const site_basis<I,STATE>& b, int s) 
{
  return is_fermionic(b.basis(),b[s]);
}


// ------------------------------- implementation ----------------------------------

template <class I, class STATE>
typename site_basis<I,STATE>::size_type site_basis<I,STATE>::index(const value_type& x) const
{
  const_iterator it = std::lower_bound(super_type::begin(),super_type::end(),x);
  return (it != super_type::end() && *it==x ? it-super_type::begin() : super_type::size());
}

template <class I, class STATE>
bool site_basis<I,STATE>::check_sort() const
{
  for (std::size_t i=0;i<super_type::size()-1;++i)
    if ((*this)[i]>=(*this)[i+1])
      return false;
  return true;
}

template <class I, class STATE>
site_basis<I,STATE>::site_basis(const SiteBasisDescriptor<I>& b)
 : basis_(b)
{
  if ((I)(b.num_states())==std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION ())
    boost::throw_exception(std::runtime_error("Cannot build infinite set of basis states\n"));
  std::stack<std::pair<typename SiteBasisDescriptor<I>::const_iterator,half_integer<I> > > s;
  typename SiteBasisDescriptor<I>::const_iterator it=b.begin();
  std::vector<half_integer<I> > quantumnumbers(basis_.size());
  const_cast<QuantumNumberDescriptor<I>&>(*it).set_parameters(b.get_parameters(true));
  for(half_integer<I> q=it->max BOOST_PREVENT_MACRO_SUBSTITUTION ();q>=it->min BOOST_PREVENT_MACRO_SUBSTITUTION ();--q) 
    s.push(std::make_pair(it,q));
  while(!s.empty()) {
    it=s.top().first;
    quantumnumbers[it-b.begin()]=s.top().second;
    s.pop();
    if(it==b.end()-1) 
      std::vector<STATE>::push_back(state_type(quantumnumbers));
    else {
      ++it;
      Parameters p=b.get_parameters(true);
      for(typename SiteBasisDescriptor<I>::const_iterator qit=b.begin();qit!=it;++qit)
        p[qit->name()]=quantumnumbers[qit-b.begin()];
      const_cast<QuantumNumberDescriptor<I>&>(*it).set_parameters(p);
      for(half_integer<I> q=it->max BOOST_PREVENT_MACRO_SUBSTITUTION ();q>=it->min BOOST_PREVENT_MACRO_SUBSTITUTION ();--q)
        s.push(std::make_pair(it,q));
    }
  }
  if(!check_sort())
    boost::throw_exception(std::logic_error("Site basis not sorted correctly"));
}

}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I, class STATE>
std::ostream& operator<<(std::ostream& out, const alps::site_basis<I,STATE>& s)
{ 
  out << "{\n";
  for (typename alps::site_basis<I,STATE>::const_iterator it=s.begin();it!=s.end();++it) {
    out << "  |";
    for (int i=0;i<s.basis().size();++i)
      out << " " << s.basis()[i].name() << "=" << get_quantumnumber(*it,i);
    out << " >\n";
  }
  out << "}\n";
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
