/***************************************************************************
* PALM++/alea library
*
* alea/observableset.C     Monte Carlo observable class
*
* $Id$
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

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

Factory::Factory()
{
  ObservableSet::register_type<IntObsevaluator>();
  ObservableSet::register_type<IntObservable>();
  ObservableSet::register_type<IntTimeSeriesObservable>();
  ObservableSet::register_type<SimpleIntObservable>();
  ObservableSet::register_type<RealObsevaluator>();
  ObservableSet::register_type<RealObservable>();
  ObservableSet::register_type<RealTimeSeriesObservable>();
  ObservableSet::register_type<SimpleRealObservable>();
#ifdef ALPS_HAVE_VALARRAY
  ObservableSet::register_type<RealVectorObsevaluator>();
  ObservableSet::register_type<RealVectorObservable>();
  ObservableSet::register_type<RealVectorTimeSeriesObservable>();
  ObservableSet::register_type<SimpleRealVectorObservable>();
  ObservableSet::register_type<IntVectorObsevaluator>();
  ObservableSet::register_type<IntVectorObservable>();
  ObservableSet::register_type<IntVectorTimeSeriesObservable>();
  ObservableSet::register_type<SimpleIntVectorObservable>();
#endif
  ObservableSet::register_type<Real2DArrayObservable>();
  ObservableSet::register_type<SimpleReal2DArrayObservable>();
  ObservableSet::register_type<HistogramObservable<int32_t> >();
  ObservableSet::register_type<HistogramObservable<int32_t,double> >();
}

Factory::~Factory()
{
  for (iterator it = begin(); it!=end(); ++it)
  {
#ifndef ALPS_NO_DELETE
    delete it->second;
#endif
  }
} 

} // namespace detail

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
    if (factory_.find(v) != factory_.end() && factory_.find(v)->second != 0) {
      Observable* obs = factory_[v]->make("untitled");
      dump >> *obs;
      addObservable(obs);
    } else {
      boost::throw_exception(std::runtime_error("No factory exists for observable type"
      +(boost::lexical_cast<std::string,uint32_t>(v))));
    }
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
#ifndef ALPS_NO_DELETE
  do_for_all(detail::deleteit);
#endif
  erase(begin(),end());  
  for (const_iterator it = m.begin(); it != m.end(); ++it)
    addObservable(it->second->clone());
  return *this;
}

ObservableSet::~ObservableSet()
{
#ifndef ALPS_NO_DELETE
  do_for_all(detail::deleteit);
#endif
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
#ifndef BOOST_NO_VOID_RETURNS
  do_for_all(boost::bind2nd(boost::mem_fun_ref(&Observable::reset),why));
#else
  do_for_all(boost::bind2nd_void(boost::mem_fun_ref(&Observable::reset),why));
#endif
}

void ObservableSet::addObservable(Observable* obs)
{
  if (obs) {
  // store pointer
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

detail::Factory ObservableSet::factory_;

void ObservableSet::compact()
{
  do_for_all(boost::mem_fun_ref(&Observable::compact));
}

void ObservableSet::write_xml(std::ostream& xml, const boost::filesystem::path& fn_hdf5) const
{
  xml << "<AVERAGES>\n";
  for(const_iterator i=begin();i!=end();i++) 
    i->second->write_xml(xml,fn_hdf5);
  xml << "</AVERAGES>\n";
}

void ObservableSet::write_xml(oxstream& oxs, const boost::filesystem::path& fn_hdf5) const
{
  oxs << alps::start_tag("AVERAGES");
  for(const_iterator i = begin(); i != end(); ++i) 
    i->second->write_xml(oxs, fn_hdf5);
  oxs << alps::end_tag("AVERAGES");
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
