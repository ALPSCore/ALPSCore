/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2008 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include <alps/alea/observable.h>

namespace alps {

Observable::~Observable() {}

Observable::Observable(const std::string& n) 
  : name_(n)
  , in_observable_set_(false) 
{
}


Observable::Observable(const Observable& o) 
  : name_(o.name_)
  , in_observable_set_(false) 
{
}

ALPS_DUMMY_VOID Observable::reset(bool equilibrated)
{
}

ALPS_DUMMY_VOID Observable::output(std::ostream&) const
{
}

Observable* Observable::clone() const
{
  return new Observable(*this);
}

uint32_t Observable::version_id() const
{
  return 0;
}


const std::string& Observable::name() const
{
  return name_;
}

void Observable::write_xml(oxstream& oxs, const boost::filesystem::path&) const
{
  oxs << start_tag("AVERAGE") << attribute("name", name()) << end_tag("AVERAGE");
}

#ifndef ALPS_WITHOUT_OSIRIS

void Observable::load(IDump& dump)
{ dump >> name_; }

void Observable::save(ODump& dump) const
{ dump << name_; }

#endif // !ALPS_WITHOUT_OSIRIS

void Observable::save(hdf5::archive &) const {}

void Observable::load(hdf5::archive &) {}

bool Observable::is_signed() const
{ return false;        }

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

const Observable& Observable::signed_observable() const
{
  if(signed())
    boost::throw_exception(std::logic_error("alps::Observable::signed_observable not implemented."));
  else
    boost::throw_exception(std::logic_error("alps::Observable::signed_observable called for unsigned Observable"));
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

void Observable::rename(const std::string& newname)
{
  name_=newname;
}

} // namespace alps

