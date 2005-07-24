/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef OSIRIS_PROCESS_H
#define OSIRIS_PROCESS_H

#include <alps/config.h>
#include <alps/osiris/std/vector.h>
#include <alps/osiris/dump.h>

#include <cstdlib>
#include <string>

namespace alps {

/** a host descriptor.
    the Host class describes a host computer. */
class Host
{
public:        
  /// construct an invalid hosts
  Host(); 
  
  /** construct a Host object.
      @param id the host id
      @param name the host name
      @param speed relative speed of the host */
  Host(int32_t id, const std::string& name="", double speed=1.0); // new Host

  /// deserialize a host descriptor.
  void load(IDump&); // load from the dump
  /// serialize a host descriptor.
  void save(ODump&) const; // save to the dump
  
  /// are two hosts the same? 
  bool operator==(const Host& h) const {return id_==h.id_;}
  /// are two hosts different? 
  bool operator!=(const Host& h) const {return id_!=h.id_;}
  
  /// does this object refer to a valid host?
  bool valid() const; // is this a valid Host descriptor ?
  
  /// the host name
  const std::string& name() const {return name_;}
  
  /// the host id
  operator int32_t () const {return id_;}

  /// relative host speed
  double speed() const {return speed_;}
 
protected:
  std::string name_; // name of the Host
  double speed_; // relative speed of the Host
  int32_t id_; // integral unique identification number

private:
 };
 
 
/** a process descriptor.
    Describes a process. Is derived from a Host class, refering to the
    host on which the process is executed. */
    
class Process : public Host
{
public:
  
  // CONSTRUCTORS
  
  Process(const Host&, int32_t); // constructor for existing processes
  explicit Process(int32_t); // constructor for process on unknown host
  Process() : Host() {} // invalid Process
  
  // MEMBER FUNCTIONS
  
  void load(IDump&); // load from a dump
  void save(ODump&) const; // save into a dump

  bool valid() const; // is this a valid Process descriptor ?

  inline bool on_host(const Host& h) const {return h==*this;}

  bool local() const; // is this the current Process?
  
  inline operator int32_t () const {return tid;}
  
  bool operator==(const Process& p) const
  { return (tid==p.tid);}

  bool operator!=(const Process& p)  const
  { return (tid!=p.tid);}

  /// sorting criterion for two processes?
  bool operator<(const Process& p) const
  { return (tid<p.tid);}

private:
  int32_t tid; // the unique Process id
};

//=======================================================================
// is_on_host and hosts_process
//
// predicates to find processes on a certain hosts and vice versa
//-----------------------------------------------------------------------

class is_on_host {
  const Host& the_host_;
public:
  is_on_host(const Host& h) : the_host_(h) {}
  bool operator()(const Process& p) const {return p.on_host(the_host_);}
};

class hosts_process {
  const Process& the_process_;
public:
  hosts_process(const Process& p) : the_process_(p) {}
  bool operator()(const Host& h) const {return the_process_.on_host(h);}
};


typedef std::vector<Host> HostList;
typedef std::vector<Process> ProcessList;
}


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline std::ostream& operator<<(std::ostream& out, const alps::Host& h)
{
  out << h.name();
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const alps::Process& p)
{
  out << "#" << int32_t(p);
  if(p.name().size()!=0)
  out << " on Host " << p.name();
  return out;
}

inline alps::ODump& operator<<(alps::ODump& od, const alps::Process& p)
{ p.save(od); return od; }

inline alps::IDump& operator>>(alps::IDump& id, alps::Process& p)
{ p.load(id); return id; }

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_PROCESS_H
