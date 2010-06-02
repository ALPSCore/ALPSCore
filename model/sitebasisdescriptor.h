/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2009 by Matthias Troyer <troyer@comp-phys.org>,
*                            Axel Grzesik <axel@th.physik.uni-bonn.de>,
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

#ifndef ALPS_MODEL_SITEBASISDESCRIPTOR_H
#define ALPS_MODEL_SITEBASISDESCRIPTOR_H

#include <alps/model/quantumnumber.h>
#include <alps/model/operatordescriptor.h>
#include <boost/config.hpp>
#include <cstddef>
#include <stack>
#include <utility>
#include <vector>
#include <iostream>

namespace alps {

template<class I>
class SiteBasisDescriptor : public std::vector<QuantumNumberDescriptor<I> >
{
  typedef std::vector<QuantumNumberDescriptor<I> > super_type;
public:
  typedef typename std::vector<QuantumNumberDescriptor<I> >::const_iterator
    const_iterator;
  typedef typename std::vector<QuantumNumberDescriptor<I> >::iterator
    iterator;
  typedef typename OperatorDescriptor<I>::operator_map operator_map;
  typedef typename operator_map::const_iterator operator_iterator;

  SiteBasisDescriptor() : num_states_(0) { }
  SiteBasisDescriptor(const std::string& name,
                      const Parameters& parms = Parameters(),
                      const operator_map& ops = operator_map())
    : parms_(parms), read_parms_(parms), name_(name), num_states_(0),operators_(ops) { }
  SiteBasisDescriptor(const XMLTag&, std::istream&);
  void write_xml(oxstream&) const;

  const std::string& name() const { return name_;}
  bool valid(const std::vector<half_integer<I> >&) const;
  std::size_t num_states() const
  {
    if (!valid_ && !const_cast<SiteBasisDescriptor*>(this)->evaluate())
      boost::throw_exception(std::runtime_error("Cannot evaluate quantum"
        " numbers in site basis " + name()));
    return num_states_;
  }
  bool set_parameters(const Parameters&, bool=false);
  const Parameters& get_parameters(bool all=false) const { return all ? parms_ : read_parms_; }
  const operator_map& operators() const { return operators_;}
  bool has_operator(const std::string& name) const
  { return operators_.find(name) != operators_.end(); }
  
  template <class STATE, class T>
  boost::tuple<STATE, expression::Expression<T>,bool> apply(const std::string& name, STATE state, const expression::ParameterEvaluator<T>& eval, bool) const;
  bool is_fermionic(const std::string& name) const;
private:
  mutable bool valid_;
  bool evaluate();
  Parameters parms_;
  Parameters read_parms_;
  std::string name_;
  mutable std::size_t num_states_;
  void init_dependencies();
  operator_map operators_;
};

// ------------------------------- implementation ----------------------------------

template <class I>
bool SiteBasisDescriptor<I>::is_fermionic(const std::string& name) const 
{
  operator_iterator op=operators_.find(name);
  if(op==operators_.end())
      return false;
  return op->second.is_fermionic(*this);
}

template <class I>
template <class STATE, class T>
boost::tuple<STATE, expression::Expression<T>,bool> 
SiteBasisDescriptor<I>::apply(const std::string& name, STATE state, const expression::ParameterEvaluator<T>& eval, bool isarg) const 
{
  operator_iterator op=operators_.find(name);
  if(op==operators_.end())
    return boost::make_tuple(state,expression::Expression<T>(),false);
  return op->second.apply(state,*this,eval,isarg);
}

template <class I>
bool SiteBasisDescriptor<I>::valid(const std::vector<half_integer<I> >& x) const
{
  alps::Parameters p(parms_);
  if(!valid_ && !const_cast<SiteBasisDescriptor*>(this)->evaluate())
    boost::throw_exception(std::runtime_error("Cannot evaluate quantum numbers in site basis " +name()));
  if (this->size() != x.size())
    return false;
  for (std::size_t i=0;i<this->size();++i) {
    const_cast<SiteBasisDescriptor<I>&>(*this)[i].set_parameters(p);
    if (!(*this)[i].valid(x[i]))
      return false;
    else
      p[(*this)[i].name()]=x[i];
  }
  return true;
}

template <class I>
bool SiteBasisDescriptor<I>::set_parameters(const Parameters& p, bool override)
{
  for (Parameters::const_iterator it=p.begin();it!=p.end();++it) {
    parms_[it->key()] = it->value();
    if (override)
      read_parms_[it->key()] = it->value();
  }
  evaluate();
  
  return valid_;
}

template <class I>
bool SiteBasisDescriptor<I>::evaluate()
{
  valid_=true;
  Parameters q_parms_(parms_);
  for (iterator it=this->begin();it!=this->end();++it) {
    it->reset_limits();
    valid_ = valid_ && it->set_parameters(q_parms_);
    if(!valid_) break;
    q_parms_[it->name()]=it->min BOOST_PREVENT_MACRO_SUBSTITUTION ();
  }
  if (valid_ && this->begin()!=this->end()) {
    num_states_=1;
    iterator rit=this->end()-1;
    // no dependencies
    while(rit->set_parameters(parms_)) {
      if(rit->levels()>=half_integer<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION ().to_double()) {
        num_states_=std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
        return true;
      }
      num_states_ *= rit->levels();
      if(rit==this->begin()) break;
      --rit;
    }
    // there are dependencies!
    if( rit!=this->begin() ) {
      unsigned int n=0;
      typedef std::pair<iterator,Parameters> q_pair;
      std::stack<q_pair> s;
      iterator it=this->begin();
      Parameters p=q_parms_;
      it->set_parameters(p);
      if(it->levels()==std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION ()) {
        num_states_=std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
        return true;
      }
      for(half_integer<I> q=it->min BOOST_PREVENT_MACRO_SUBSTITUTION ();q<=it->max BOOST_PREVENT_MACRO_SUBSTITUTION ();++q) {
        p[it->name()]=q;
        s.push(q_pair(it,p));
      }
      while(!s.empty()) {
        iterator it=s.top().first;
        Parameters      p=s.top().second;
        s.pop();
        iterator itt=it+1;
        if(itt==rit) {
          itt->set_parameters(p);
          if(itt->levels()==std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION ()) {
            num_states_=std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
            return true;
          }
          n+=itt->levels();
        }
        else {
          ++it;
          it->set_parameters(p);
          if(it->levels()==std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION ()) {
            num_states_=std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
            return true;
          }
          for(half_integer<I> q=it->min BOOST_PREVENT_MACRO_SUBSTITUTION ();q<=it->max BOOST_PREVENT_MACRO_SUBSTITUTION ();++q) {
            p[it->name()]=q;
            s.push(q_pair(it,p));
          }
        }
      }
      num_states_ *= n;
    }
  }
  return valid_;
}

template <class I>
SiteBasisDescriptor<I>::SiteBasisDescriptor(const XMLTag& intag, std::istream& is)
 : valid_(false)
{
  XMLTag tag(intag);
  name_ = tag.attributes["name"];
  if (tag.type!=XMLTag::SINGLE) {
    tag = parse_tag(is);
    if (tag.name!="/SITEBASIS") {
      while (tag.name=="PARAMETER") {
        read_parms_[tag.attributes["name"]]=tag.attributes["default"];
        if (tag.type!=XMLTag::SINGLE)
          tag = parse_tag(is);
        tag = parse_tag(is);
      }
      while (tag.name=="QUANTUMNUMBER") {
        std::vector<QuantumNumberDescriptor<I> >::push_back(QuantumNumberDescriptor<I>(tag,is));
        if (tag.type!=XMLTag::SINGLE)
          tag = parse_tag(is);
        tag = parse_tag(is);
      }
      while (tag.name=="OPERATOR") {
        operators_[tag.attributes["name"]] = OperatorDescriptor<I>(tag,is);
        tag = parse_tag(is);
      }
    }
    if (tag.name !="/SITEBASIS")
      boost::throw_exception(std::runtime_error("Illegal tag <" + tag.name + "> in <SITEBASIS> element"));
  }
  parms_=read_parms_;
  init_dependencies();
  // I need this line, otherwise the expressions in quantumnumbers cannot be evaluated. Dirty patch. Looks like a bug. To be looked at again. Axel Grzesik, 07/08/03
  evaluate();
}

template<class I>
void SiteBasisDescriptor<I>::init_dependencies() {
  for(iterator it=this->begin();it!=this->end();++it)
    for(iterator jt=this->begin();jt!=it;++jt)
      if(it->depends_on(jt->name()))
        it->add_dependency(*jt);
}

template <class I>
void SiteBasisDescriptor<I>::write_xml(oxstream& os) const
{
  os << start_tag("SITEBASIS") << attribute("name", name());
  for (Parameters::const_iterator it=read_parms_.begin();it!=read_parms_.end();++it)
    os << start_tag("PARAMETER") << attribute("name", it->key())
       << attribute("default", it->value()) << end_tag("PARAMETER");
  for (const_iterator it=this->begin();it!=this->end();++it)
    os << *it;
  for (typename operator_map::const_iterator it=operators_.begin();it!=operators_.end();++it)
    os << it->second;
  os << end_tag("SITEBASIS");
}

}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I>
inline alps::oxstream& operator<<(alps::oxstream& out, const alps::SiteBasisDescriptor<I>& q)
{
  q.write_xml(out);
  return out;
}

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::SiteBasisDescriptor<I>& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
