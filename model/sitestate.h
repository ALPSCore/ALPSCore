/***************************************************************************
* ALPS++/model library
*
* model/sitestate.h    the state on a single site
*
* $Id$
*
* Copyright (C) 2003-2003 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_MODEL_SITESTATE_H
#define ALPS_MODEL_SITESTATE_H

#include <alps/model/sitebasisdescriptor.h>
#include <vector>
#include <iostream>

namespace alps {

template <class I>
class StateDescriptor : public std::vector<half_integer<I> > {
public:
  typedef half_integer<I> quantumnumber_type;
  typedef typename std::vector<half_integer<I> >::const_iterator const_iterator;
  StateDescriptor() {}
  StateDescriptor(const std::vector<half_integer<I> >& x) : std::vector<half_integer<I> >(x)  {}
};


template <class I>
class SingleQNStateDescriptor {
public:
  typedef half_integer<I> representation_type;
  typedef half_integer<I> quantumnumber_type;
  typedef std::size_t size_type;
  
  SingleQNStateDescriptor() {}
  SingleQNStateDescriptor(representation_type x) : state_(x)  {}
  template <class J>
  SingleQNStateDescriptor(const std::vector<half_integer<J> >& x) { assert(x.size()==1); state_=x[0];}
  operator representation_type() const { return state_;}
  representation_type state() const { return state_;}
  representation_type& state() { return state_;}
private:
  representation_type state_;
};


template <class I>
bool operator < (const SingleQNStateDescriptor<I>& x,  const SingleQNStateDescriptor<I>& y)
{
  return x.state() < y.state();
}

template <class I>
bool operator > (const SingleQNStateDescriptor<I>& x,  const SingleQNStateDescriptor<I>& y)
{
  return x.state() > y.state();
}

template <class I>
bool operator == (const SingleQNStateDescriptor<I>& x,  const SingleQNStateDescriptor<I>& y)
{
  return x.state() == y.state();
}

template <class I>
bool operator <= (const SingleQNStateDescriptor<I>& x,  const SingleQNStateDescriptor<I>& y)
{
  return x.state() <= y.state();
}

template <class I>
bool operator >= (const SingleQNStateDescriptor<I>& x,  const SingleQNStateDescriptor<I>& y)
{
  return x.state() >= y.state();
}


template <class I>
half_integer<I> get_quantumnumber(const StateDescriptor<I>& s, typename StateDescriptor<I>::size_type i)
{
  if (i>=s.size())
    boost::throw_exception(std::logic_error("Called get_quantumnumber with illegal index"));
  return s[i];
}

template <class I>
half_integer<I> get_quantumnumber(const SingleQNStateDescriptor<I>& s, std::size_t i)
{
  if (i!=0)
    boost::throw_exception(std::logic_error("Called get_quantumnumber with illegal index"));
  return s.state();
}

template <class I>
half_integer<I>& get_quantumnumber(StateDescriptor<I>& s, typename StateDescriptor<I>::size_type i)
{
  if (i>=s.size())
    boost::throw_exception(std::logic_error("Called get_quantumnumber with illegal index"));
  return s[i];
}

template <class I>
half_integer<I>& get_quantumnumber(SingleQNStateDescriptor<I>& s, std::size_t i)
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

}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I>
std::ostream& operator<<(std::ostream& out, const alps::StateDescriptor<I>& s)
{
  out << "|";
  for (typename alps::StateDescriptor<I>::const_iterator it=s.begin();it!=s.end();++it)
    out << *it << " ";
  out << ">";
  return out;        
}

template <class I>
std::ostream& operator<<(std::ostream& out, const alps::SingleQNStateDescriptor<I>& s)
{
  out << "|" << s.state() << ">";
  return out;        
}


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
