/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

#include <alps/scheduler/info.h>
#include <alps/osiris/std/vector.h>
#include <alps/osiris/os.h>

#include <algorithm>
#include <boost/functional.hpp>

namespace alps {
namespace scheduler {

Info::Info()
  : startt(boost::posix_time::second_clock::local_time()),
    stopt(boost::posix_time::second_clock::local_time()),
    host(alps::hostname())
{
}

void Info::save(ODump& dump) const
{
  dump << host << boost::posix_time::to_iso_string(startt) 
       << boost::posix_time::to_iso_string(stopt) << phase;
}

void Info::load(IDump& dump, int version)
{
  dump >> host;
  if (version<300) {
    startt = boost::posix_time::from_time_t(int32_t(dump));
    stopt = boost::posix_time::from_time_t(int32_t(dump));
  }
  else {
    std::string tmp;
    dump >> tmp;
    startt = boost::posix_time::from_iso_string(tmp);
    dump >> tmp;
    stopt = boost::posix_time::from_iso_string(tmp);
  }
  if (version<200) {
    int32_t reason;
    int32_t thermalized;
    dump >> reason >> thermalized;
    switch(reason) {
      case THERMALIZED:
        phase = "equilibrating";
        break;
      case INTERRUPTED:
      case HALTED:
        phase = "running";
        break;
      default:
        boost::throw_exception(std::logic_error("unknow reason in Info::load"));
    }
  }
  else
    dump >> phase; 
}


// start the run: save start time and current time as last checkpoint
void Info::start(const std::string& p)
{
  startt = stopt = boost::posix_time::second_clock::local_time();
  phase = p;
}


// halt the run: save current time as stop time
void Info::halt()
{
  stopt=boost::posix_time::second_clock::local_time();
}


// make a checkpoint: store current time as stop time
void Info::checkpoint()
{
  stopt=boost::posix_time::second_clock::local_time();
}


ALPS_DUMMY_VOID Info::write_xml(alps::oxstream& xml) const
{
  xml << start_tag("EXECUTED");
  if (phase!="")
    xml << attribute("phase",phase);
  xml << start_tag("FROM") << no_linebreak << boost::posix_time::to_simple_string(startt) << end_tag("FROM");
  xml << start_tag("TO") << no_linebreak << boost::posix_time::to_simple_string(stopt) << end_tag("TO");
  xml << start_tag("MACHINE") << no_linebreak << start_tag("NAME") << host 
      << end_tag("NAME") << end_tag("MACHINE");
  xml << end_tag("EXECUTED");
  ALPS_RETURN_VOID
}


void TaskInfo::save (ODump& dump) const
{
  if(!empty()) // update checkpoint time if running
    const_cast<TaskInfo&>(*this).rbegin()->checkpoint();
  dump << static_cast<const std::vector<Info>&>(*this);
}


void TaskInfo::load(IDump& dump, int version)
{
  resize(static_cast<int32_t>(dump));
  for (int i=0;i<size();++i)
    operator[](i).load(dump,version);
  if (version<200) {
    std::string host;
    int32_t dummy;
    dump >> host >> dummy;
    if(dummy)
      dump >> dummy;
    int find_thermalized=0;
    for (int i=0; i<size();++i)
      if (at(i).phase=="equlibrating")
        find_thermalized=i;
    for (int i=0;i<find_thermalized;++i)
      at(i).phase="equlibrating";
  }
}


// start the run new: create new info
void TaskInfo::start(const std::string& phase)
{
  push_back(Info());
  rbegin()->start(phase);
}


// halt the run: store times
void TaskInfo::halt()
{
  if(!empty())
    rbegin()->halt();
  else
    boost::throw_exception( std::logic_error("empty TaskInfo in TaskInfo::halt"));
}


void TaskInfo::write_xml(alps::oxstream& xml) const
{
  std::for_each(begin(),end(),boost::bind2nd(boost::mem_fun_ref(&Info::write_xml),xml));
}

} // namespace scheduler
} // namespace alps
