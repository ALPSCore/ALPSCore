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

#include <alps/scheduler/info.h>
#include <alps/osiris/std/vector.h>
#include <alps/utility/os.hpp>

#include <algorithm>
#include <boost/functional.hpp>

namespace alps {
namespace scheduler {

Info::Info()
  : startt_(boost::posix_time::second_clock::local_time()),
    stopt_(boost::posix_time::second_clock::local_time()),
    host_(alps::hostname())
{
}

#ifdef ALPS_HAVE_HDF5
void Info::save(hdf5::archive & ar) const {
  ar
      << make_pvp("machine/name", host_)
      << make_pvp("from", boost::posix_time::to_iso_string(startt_))
      << make_pvp("to", boost::posix_time::to_iso_string(stopt_))
      << make_pvp("phase", phase_)
  ;
}
void Info::load(hdf5::archive & ar) {
  std::string startt, stopt;
  ar
      >> make_pvp("from", startt)
      >> make_pvp("to", stopt)
      >> make_pvp("machine/name", host_)
      >> make_pvp("phase", phase_)
  ;
  startt_ = boost::posix_time::from_iso_string(startt);
  stopt_ = boost::posix_time::from_iso_string(stopt);
}
#endif

void Info::save(ODump& dump) const
{
  dump << host_ << boost::posix_time::to_iso_string(startt_) 
       << boost::posix_time::to_iso_string(stopt_) << phase_;
}

void Info::load(IDump& dump, int version)
{
  dump >> host_;
  if (version<300) {
    startt_ = boost::posix_time::from_time_t(int32_t(dump));
    stopt_ = boost::posix_time::from_time_t(int32_t(dump));
  }
  else {
    std::string tmp;
    dump >> tmp;
    startt_ = boost::posix_time::from_iso_string(tmp);
    dump >> tmp;
    stopt_ = boost::posix_time::from_iso_string(tmp);
  }
  if (version<200) {
    int32_t reason;
    int32_t thermalized;
    dump >> reason >> thermalized;
    switch(reason) {
      case THERMALIZED:
        phase_ = "equilibrating";
        break;
      case INTERRUPTED:
      case HALTED:
        phase_ = "running";
        break;
      default:
        boost::throw_exception(std::logic_error("unknow reason in Info::load"));
    }
  }
  else
    dump >> phase_; 
}


// start the run: save start time and current time as last checkpoint
void Info::start(const std::string& p)
{
  startt_ = stopt_ = boost::posix_time::second_clock::local_time();
  phase_ = p;
}


// halt the run: save current time as stop time
void Info::halt()
{
  stopt_=boost::posix_time::second_clock::local_time();
}


// make a checkpoint: store current time as stop time
void Info::checkpoint()
{
  stopt_=boost::posix_time::second_clock::local_time();
}

const boost::posix_time::ptime& Info::start_time() const { return startt_; }

const boost::posix_time::ptime& Info::stop_time() const { return stopt_; }

const std::string& Info::phase() const { return phase_; }

const std::string& Info::host() const { return host_; }

ALPS_DUMMY_VOID Info::write_xml(alps::oxstream& xml) const
{
  xml << start_tag("EXECUTED");
  if (phase_!="")
    xml << attribute("phase",phase_);
  xml << start_tag("FROM") << no_linebreak << boost::posix_time::to_simple_string(startt_) << end_tag("FROM");
  xml << start_tag("TO") << no_linebreak << boost::posix_time::to_simple_string(stopt_) << end_tag("TO");
  xml << start_tag("MACHINE") << no_linebreak << start_tag("NAME") << host_ 
      << end_tag("NAME") << end_tag("MACHINE");
  xml << end_tag("EXECUTED");
  ALPS_RETURN_VOID
}

#ifdef ALPS_HAVE_HDF5
void TaskInfo::save(hdf5::archive & ar) const {
  if(!empty())
    const_cast<TaskInfo &>(*this).back().checkpoint();
  for (unsigned int i=0 ; i < size() ; ++i)
    ar << make_pvp(boost::lexical_cast<std::string>(i), (*this)[i]);
}
void TaskInfo::load(hdf5::archive & ar) {
    std::vector<std::string> list = ar.list_children("/log/alps");
    resize(list.size());
    for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it)
      ar >> make_pvp(*it, (*this)[it - list.begin()]);
}
#endif

void TaskInfo::save (ODump& dump) const
{
  if(!empty()) // update checkpoint time if running
    const_cast<TaskInfo&>(*this).rbegin()->checkpoint();
  dump << static_cast<const std::vector<Info>&>(*this);
}


void TaskInfo::load(IDump& dump, int version)
{
  resize(static_cast<int32_t>(dump));
  for (unsigned int i=0;i<size();++i)
    operator[](i).load(dump,version);
  if (version<200) {
    std::string host_;
    int32_t dummy;
    dump >> host_ >> dummy;
    if(dummy)
      dump >> dummy;
    int find_thermalized=0;
    for (unsigned int i=0; i<size();++i)
      if (at(i).phase_=="equlibrating")
        find_thermalized=i;
    for (int i=0;i<find_thermalized;++i)
      at(i).phase_="equlibrating";
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
