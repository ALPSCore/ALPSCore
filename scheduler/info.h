/***************************************************************************
* ALPS++/scheduler library
*
* scheduler/info.h   A class to store parameters
*
* $Id$
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_SCHEDULER_INFO_H
#define ALPS_SCHEDULER_INFO_H

#include <alps/scheduler/types.h>
#include <alps/osiris.h>
#include <iterator>
#include <ctime>

namespace alps {
namespace scheduler {

//=======================================================================
// Info
//
// information about a specific computation on a worker
//-----------------------------------------------------------------------

class TaskInfo;

class Info 
{
  friend class TaskInfo;
public:
  Info();
  void start(const std::string&); // register that it is started/restarted NOW
  void halt(); // register that it is halted/thermalized NOW
  void checkpoint(); // we are checkpointing, update info beforehand
  
  // write the info
  void save (ODump&) const;
  void write_xml(std::ostream&) const;
  void load (IDump& dump,int version=MCDump_task_version);
private:
   // how was it stopped? ... for historic reasons
  enum { HALTED=1, INTERRUPTED=2, THERMALIZED=3, NOTSTARTED=4 };
  time_t startt; // start time
  time_t stopt; // stop time
  std::string phase; // what was  done?
  std::string host; // which host is it running on?
};


class TaskInfo : public std::vector<Info> 
{
public:
  TaskInfo() {}

  void start(const std::string&); // the run is started/restarted NOW
  void halt(); // the run is halted/thermalized NOW
  
  void save (ODump& dump) const;
  void load (IDump& dump,int version=MCDump_task_version);
  void write_xml(std::ostream&) const;
};

} // end namespace scheduler
} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
namespace scheduler {
#endif

inline std::ostream& operator<<(std::ostream& o,const alps::scheduler::Info& i)
{
  i.write_xml(o);
  return o;
}

inline alps::IDump& operator>>(alps::IDump& dump, alps::scheduler::Info& i)
{
  i.load(dump);
  return dump;
}

inline alps::ODump& operator<< (alps::ODump& dump, const alps::scheduler::Info& info)
{
  info.save(dump);
  return dump;
}

inline std::ostream& operator<<(std::ostream& o,const alps::scheduler::TaskInfo& i)
{
  i.write_xml(o);
  return o;
}

inline alps::IDump& operator>>(alps::IDump& dump, alps::scheduler::TaskInfo& i)
{
  i.load(dump);
  return dump;
}

inline alps::ODump& operator<< (alps::ODump& dump, const alps::scheduler::TaskInfo& info)
{
  info.save(dump);
  return dump;
}


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace scheduler
} // namespace alps
#endif

#endif
