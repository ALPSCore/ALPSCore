/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2004 by Matthias Troyer <troyer@comp-phys.org>
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

#ifndef ALPS_MODEL_SITESTATE_H
#define ALPS_MODEL_SITESTATE_H

#include <alps/model/sitebasisdescriptor.h>
#include <vector>
#include <iostream>

namespace alps {

template <class I>
class site_state : public std::vector<half_integer<I> > {
public:
  typedef half_integer<I> quantumnumber_type;
  typedef typename std::vector<half_integer<I> >::const_iterator const_iterator;
  site_state() {}
  site_state(const std::vector<half_integer<I> >& x) : std::vector<half_integer<I> >(x)  {}
};


template <class I>
class single_qn_site_state {
public:
  typedef half_integer<I> representation_type;
  typedef half_integer<I> quantumnumber_type;
  typedef std::size_t size_type;
  
  single_qn_site_state() {}
  single_qn_site_state(representation_type x) : state_(x)  {}
  template <class J>
  single_qn_site_state(const std::vector<half_integer<J> >& x) { assert(x.size()==1); state_=x[0];}
  operator representation_type() const { return state_;}
  representation_type state() const { return state_;}
  representation_type& state() { return state_;}
private:
  representation_type state_;
};


template <class I>
bool operator < (const single_qn_site_state<I>& x,  const single_qn_site_state<I>& y)
{
  return x.state() < y.state();
}

template <class I>
bool operator > (const single_qn_site_state<I>& x,  const single_qn_site_state<I>& y)
{
  return x.state() > y.state();
}

template <class I>
bool operator == (const single_qn_site_state<I>& x,  const single_qn_site_state<I>& y)
{
  return x.state() == y.state();
}

template <class I>
bool operator <= (const single_qn_site_state<I>& x,  const single_qn_site_state<I>& y)
{
  return x.state() <= y.state();
}

template <class I>
bool operator >= (const single_qn_site_state<I>& x,  const single_qn_site_state<I>& y)
{
  return x.state() >= y.state();
}


template <class I>
half_integer<I> get_quantumnumber(const site_state<I>& s, typename site_state<I>::size_type i)
{
  if (i>=s.size())
    boost::throw_exception(std::logic_error("Called get_quantumnumber with illegal index"));
  return s[i];
}

template <class I>
half_integer<I> get_quantumnumber(const single_qn_site_state<I>& s, std::size_t i)
{
  if (i!=0)
    boost::throw_exception(std::logic_error("Called get_quantumnumber with illegal index"));
  return s.state();
}

template <class I>
half_integer<I>& get_quantumnumber(site_state<I>& s, typename site_state<I>::size_type i)
{
  if (i>=s.size())
    boost::throw_exception(std::logic_error("Called get_quantumnumber with illegal index"));
  return s[i];
}

template <class I>
half_integer<I>& get_quantumnumber(single_qn_site_state<I>& s, std::size_t i)
{
  if (i!=0)
    boost::throw_exception(std::logic_error("Called get_quantumnumber with illegal index"));
  return s.state();
}

template <class I>
std::size_t get_quantumnumber_index(const std::string& n, const SiteBasisDescriptor<I>& b)
{
  for (std::size_t i=0;i<b.size();++i) {
    if (b[i].name()==n)
      return i;
  }
  return b.size();
}

template <class S, class I>
typename S::quantumnumber_type get_quantumnumber(const S& s, const std::string& n, const SiteBasisDescriptor<I>& b)
{
  return get_quantumnumber(s,get_quantumnumber_index(n,b));
}

template <class I, class S>
bool is_fermionic(const SiteBasisDescriptor<I>& b, const S& s)
{
  bool f=false;
  for (std::size_t i=0;i<b.size();++i)
    if (b[i].fermionic() && is_odd(get_quantumnumber(s,i)))
      f=!f;
  return f;
}

}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I>
std::ostream& operator<<(std::ostream& out, const alps::site_state<I>& s)
{
  out << "|";
  for (typename alps::site_state<I>::const_iterator it=s.begin();it!=s.end();++it)
    out << *it << " ";
  out << ">";
  return out;        
}

template <class I>
std::ostream& operator<<(std::ostream& out, const alps::single_qn_site_state<I>& s)
{
  out << "|" << s.state() << ">";
  return out;        
}


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
