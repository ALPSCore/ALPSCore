/***************************************************************************
* ALPS++/alea library
*
* alps/alea/observable.C     Monte Carlo observable class
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

#include <alps/alea/observable.h>

namespace alps {

const std::string& Observable::name() const
{
  return name_;
}

void Observable::write_xml(std::ostream& xml,const boost::filesystem::path&) const
{
  xml << "<AVERAGE name=\"" << name() << "\"/>\n";
}

void Observable::write_xml(oxstream& oxs, const boost::filesystem::path&) const
{
  oxs << start_tag("AVERAGE") << attribute("name", name()) << end_tag;
}

#ifndef ALPS_WITHOUT_OSIRIS

void Observable::load(IDump& dump) 
{ dump >> name_; }

void Observable::save(ODump& dump) const
{ dump << name_; }

#endif // !ALPS_WITHOUT_OSIRIS

bool Observable::is_signed() const 
{ return false;	}

void Observable::set_sign(const Observable&) 
{
  if(is_signed()) 
    boost::throw_exception(std::logic_error("alps::Observable::set_sign not implemented."));
  else
    boost::throw_exception(std::logic_error("alps::Observable::set_sign called for unsigned Observable"));
}

void Observable::clear_sign() 
{
  if(signed()) 
    boost::throw_exception(std::logic_error("alps::Observable::clear_sign not implemented."));
  else
    boost::throw_exception(std::logic_error("alps::Observable::clear_sign called for unsigned Observable"));
}


void Observable::set_sign_name(const std::string&) 
{
  if(signed()) 
    boost::throw_exception(std::logic_error("alps::Observable::set_sign_name not implemented."));
  else
    boost::throw_exception(std::logic_error("alps::Observable::set_sign_name called for unsigned Observable"));
}

const std::string Observable::sign_name() const
{
  if(signed()) 
    boost::throw_exception(std::logic_error("alps::Observable::sign_name not implemented."));
  else
    boost::throw_exception(std::logic_error("alps::Observable::sign_name called for unsigned Observable"));
  return "";
}

const Observable& Observable::sign() const
{
  if(signed()) 
    boost::throw_exception(std::logic_error("alps::Observable::sign not implemented."));
  else
    boost::throw_exception(std::logic_error("alps::Observable::sign called for unsigned Observable"));
  return (*reinterpret_cast<Observable*>(1));
}

uint32_t Observable::number_of_runs() const 
{
  return 1;
}

Observable* Observable::get_run(uint32_t) const 
{
  return clone();
}

bool Observable::can_merge() const
{
  return false;
}

bool Observable::can_merge(const Observable&) const
{
  return false;
}

void Observable::merge(const Observable&)
{
  boost::throw_exception(std::logic_error("alps::Observable::merge not implemented."));
}

Observable* Observable::convert_mergeable() const
{
  boost::throw_exception(std::logic_error("alps::Observable::convert_mergeable not implemented."));
  return 0;
}
  
bool Observable::can_set_thermalization() const
{
  return false;
}

void Observable::set_thermalization(uint32_t)
{
  boost::throw_exception(std::logic_error("alps::Observable::set_thermalization not implemented for this observable."));
}

void Observable::compact()
{
  // do nothing
}

void Observable::rename(const std::string& newname)
{
  if (in_observable_set_)
    boost::throw_exception(std::runtime_error("Cannot change name of an Observable in an ObservableSet."));
  name_=newname;
}

} // namespace alps

