/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include <alps/alea/observableset.h>
#include <alps/alea/simpleobservable.h>
#include <alps/alea/nobinning.h>
#include <alps/alea/detailedbinning.h>
#include <alps/alea/simpleobseval.h>
#include <alps/alea/histogram.h>
#include <alps/multi_array.hpp>
#include <boost/lexical_cast.hpp>

namespace alps {

namespace detail {

inline void deleteit(Observable& obs)
{
  delete &obs;
}

} // end namespace detail



ObservableFactory::ObservableFactory()
{
  register_observable<IntObsevaluator>();
  register_observable<IntObservable>();
  register_observable<IntTimeSeriesObservable>();
  register_observable<SimpleIntObservable>();
  register_observable<RealObsevaluator>();
  register_observable<RealObservable>();
  register_observable<RealTimeSeriesObservable>();
  register_observable<SimpleRealObservable>();
#ifdef ALPS_HAVE_VALARRAY
  register_observable<RealVectorObsevaluator>();
  register_observable<RealVectorObservable>();
  register_observable<RealVectorTimeSeriesObservable>();
  register_observable<SimpleRealVectorObservable>();
  register_observable<IntVectorObsevaluator>();
  register_observable<IntVectorObservable>();
  register_observable<IntVectorTimeSeriesObservable>();
  register_observable<SimpleIntVectorObservable>();
#endif
  register_observable<Real2DArrayObservable>();
  register_observable<SimpleReal2DArrayObservable>();
  register_observable<HistogramObservable<int32_t> >();
  register_observable<HistogramObservable<int32_t,double> >();
}


#ifndef ALPS_WITHOUT_OSIRIS

void ObservableSet::save(ODump& dump) const
{
  dump << uint32_t(size());
  for (base_type::const_iterator it = begin();it != end(); ++it){
    dump << it->second->version_id();
    dump << *(it->second);
  }
}

void ObservableSet::load(IDump& dump) 
{
  uint32_t n(dump);
  for (int i = 0; i < n; ++i) {
    uint32_t v(dump);
    Observable* obs = factory_.create(v);
    dump >> *obs;
    addObservable(obs);
  }
}

#endif

void ObservableSet::update_signs()
{
  for (iterator it = begin(); it != end(); ++it)
    if(it->second->is_signed() && has(it->second->sign_name()))
      it->second->set_sign((*this)[it->second->sign_name()]);
}

ObservableSet::ObservableSet(const ObservableSet& m)
  : std::map<std::string,Observable*>()
{
  for (const_iterator it=m.begin();it!=m.end();++it) 
    addObservable(it->second->clone());
}

const ObservableSet& ObservableSet::operator=(const ObservableSet& m)
{
  do_for_all(detail::deleteit);
  erase(begin(),end());  
  for (const_iterator it = m.begin(); it != m.end(); ++it)
    addObservable(it->second->clone());
  return *this;
}

ObservableSet::~ObservableSet()
{
  do_for_all(detail::deleteit);
}

Observable& ObservableSet::operator[](const std::string& name) 
{
  base_type::iterator it = base_type::find(name);
  if(it == base_type::end()) 
    boost::throw_exception(std::out_of_range("No Observable found with the name: "+name));
  return *((*it).second);
}

const Observable& ObservableSet::operator[](const std::string& name) const 
{
  base_type::const_iterator it = base_type::find(name);
  if(it == base_type::end()) 
    boost::throw_exception(std::out_of_range("No Observable found with the name: "+name));
  return *((*it).second);
}

bool ObservableSet::has(const std::string& name) const
{
  base_type::const_iterator it = base_type::find(name);
  return it != base_type::end();
}

void ObservableSet::reset(bool why)
{
  do_for_all(boost::bind2nd(boost::mem_fun_ref(&Observable::reset),why));
}

void ObservableSet::addObservable(Observable* obs)
{
  if (obs) {
  // store pointer
  obs->clone();
  obs->added_to_set(); // can no longer change name
  if(has(obs->name()))
    removeObservable(obs->name());
  base_type::operator[](obs->name())= obs;
  
  // insert into sign list if signed and set sign if possible
  if(obs->is_signed())
  {
          signs_.insert(std::make_pair(obs->sign_name(),obs->name()));
          
    // set sign if possible
    if(has(obs->sign_name()))
      obs->set_sign((*this)[obs->sign_name()]);
  }
  
  // set where this is sign
  for (signmap::iterator it=signs_.lower_bound(obs->name());
       it != signs_.upper_bound(obs->name()); ++it)
    (*this)[it->second].set_sign(*obs);
  }
}

void ObservableSet::removeObservable(const std::string& name) 
{
  base_type::iterator it=base_type::find(name);
  if(it==base_type::end()) 
    boost::throw_exception(std::out_of_range("No Observable found with the name: "+name));

  // delete where this is sign
  for (signmap::iterator is=signs_.lower_bound(name);
    is != signs_.upper_bound(name); ++is)
    (*this)[is->second].clear_sign();
  
  if(it->second->is_signed())
  {
    // remove its sign entry
    for (signmap::iterator is=signs_.lower_bound(it->second->sign_name());
      is != signs_.upper_bound(it->second->sign_name()); ++is)
      if(is->second == name)
        signs_.erase(is);
  }
  
#ifndef ALPS_NO_DELETE
  delete it->second;
#endif
  erase(it);  
}

bool ObservableSet::can_set_thermalization_all() const
{
  bool can=true;
  for (const_iterator it=begin(); it !=end(); ++it)
    can = can && it->second->can_set_thermalization();
  return can;
}

bool ObservableSet::can_set_thermalization_any() const
{
  bool can=false;
  for (const_iterator it=begin(); it !=end(); ++it)
    can =  can || it->second->can_set_thermalization();
  return can;
}

void ObservableSet::set_thermalization(uint32_t n) 
{
  for (iterator it=begin(); it !=end(); ++it)
    if(it->second->can_set_thermalization())
      it->second->set_thermalization(n);
}

uint32_t ObservableSet::get_thermalization() const
{
  uint32_t thermal=0;
  bool got_one=false;
  for (const_iterator it=begin(); it !=end(); ++it)
    if(it->second->get_thermalization())
    {
      if(got_one)
        thermal = std::min(thermal,it->second->get_thermalization());
      else
      {
        thermal = it->second->get_thermalization();
        got_one=true;
      }
    }
  return thermal;
}

uint32_t ObservableSet::number_of_runs() const
{
  uint32_t n=0;
  for (const_iterator it=begin(); it !=end(); ++it)
    n = std::max(n, it->second->number_of_runs());
  return n;
}

ObservableSet ObservableSet::get_run(uint32_t i) const
{
  ObservableSet runset;
  for (const_iterator it=begin(); it !=end(); ++it)
    if(i<it->second->number_of_runs())
      runset.addObservable(it->second->get_run(i));
  return runset;
}

const ObservableSet& ObservableSet::operator<<(const ObservableSet& obs)
{
  for (const_iterator it=obs.begin(); it !=obs.end(); ++it)
    (*this) << *(it->second);
  return *this;
}

const ObservableSet& ObservableSet::operator<<(const Observable& obs)
{
  if(has(obs.name()))
  {
    if(!(*this)[obs.name()].can_merge())
      addObservable((*this)[obs.name()].convert_mergeable());
    (*this)[obs.name()].merge(obs);
  }
  else
    addObservable(obs.clone());
  return *this;
}

ObservableFactory ObservableSet::factory_;

void ObservableSet::compact()
{
  do_for_all(boost::mem_fun_ref(&Observable::compact));
}


void ObservableSet::write_xml(oxstream& oxs, const boost::filesystem::path& fn_hdf5) const
{
  oxs << start_tag("AVERAGES");
  for(const_iterator i = begin(); i != end(); ++i)
    i->second->write_xml(oxs, fn_hdf5);
  oxs << end_tag("AVERAGES");
}

void ObservableSet::read_xml(std::istream& infile, const XMLTag& intag)
{
  if (intag.type == XMLTag::SINGLE)
    return;
  XMLTag tag = parse_tag(infile);
  while (tag.name != "/" + intag.name) {
    if (tag.name == "SCALAR_AVERAGE")
      operator<<(RealObsevaluator(tag.attributes["name"],infile,tag));
    else if (tag.name == "VECTOR_AVERAGE")
      operator<<(RealVectorObsevaluator(tag.attributes["name"],infile,tag));
    else
      boost::throw_exception(std::runtime_error("Cannot parse tag " + tag.name + " in <" + intag.name + ">"));
    tag = parse_tag(infile);
  }
}

} // namespace alps
