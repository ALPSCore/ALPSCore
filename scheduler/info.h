/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2006 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

#ifndef ALPS_SCHEDULER_INFO_H
#define ALPS_SCHEDULER_INFO_H

// for MSVC
#if defined(_MSC_VER)
# pragma warning(disable:4275)
#endif

#include <alps/osiris/dump.h>
#include <alps/hdf5.hpp>
#include <alps/scheduler/types.h>
#include <alps/parser/xmlstream.h>

#ifdef tolower
    #undef tolower
#endif
#ifdef toupper
    #undef toupper
#endif

#include <boost/date_time/posix_time/posix_time.hpp>

#include <iterator>
#include <ctime>

namespace alps {
namespace scheduler {

//=======================================================================
// Info
//
// information about a specific computation on a worker
//-----------------------------------------------------------------------

class ALPS_DECL TaskInfo;

class ALPS_DECL Info
{
  friend class TaskInfo;
public:
  Info();
  void start(const std::string&); // register that it is started/restarted NOW
  void halt(); // register that it is halted/thermalized NOW
  void checkpoint(); // we are checkpointing, update info beforehand

  // write the info
  void save(hdf5::archive &) const;
  void load(hdf5::archive &);

  void save (ODump&) const;
  ALPS_DUMMY_VOID write_xml(alps::oxstream&) const;
  void load (IDump& dump,int version=MCDump_worker_version);

  const boost::posix_time::ptime& start_time() const;
  const boost::posix_time::ptime& stop_time() const;
  const std::string& phase() const;
  const std::string& host() const;

private:
   // how was it stopped? ... for historic reasons
  enum { HALTED=1, INTERRUPTED=2, THERMALIZED=3, NOTSTARTED=4 };
  boost::posix_time::ptime startt_; // start time
  boost::posix_time::ptime stopt_; // stop time
  std::string phase_; // what was  done?
  std::string host_; // which host is it running on?
};


class ALPS_DECL TaskInfo : public std::vector<Info>
{
public:
  TaskInfo() {}

  void start(const std::string&); // the run is started/restarted NOW
  void halt(); // the run is halted/thermalized NOW
  
  void save(hdf5::archive &) const;
  void load(hdf5::archive &);

  void save (ODump& dump) const;
  void load (IDump& dump,int version=MCDump_worker_version);
  void write_xml(alps::oxstream&) const;
};

} // end namespace scheduler
} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
namespace scheduler {
#endif

inline alps::oxstream& operator<<(alps::oxstream& o,const alps::scheduler::Info& i)
{
  i.write_xml(o);
  return o;
}

inline alps::oxstream& operator<<(alps::oxstream& o,const alps::scheduler::TaskInfo& i)
{
  i.write_xml(o);
  return o;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace scheduler
} // namespace alps
#endif

#endif
