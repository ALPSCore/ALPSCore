/***************************************************************************
* ALPS++/scheduler library
*
* scheduler/worker.C
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@comp-phys.org>,
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

#include <alps/scheduler/scheduler.h>
#include <alps/osiris.h>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem/operations.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>

namespace alps {
namespace scheduler {

//=======================================================================
// Worker
//
// base class for the actual run class which will be implemented by the
// user.
//-----------------------------------------------------------------------

Worker::Worker(const ProcessList& w,const alps::Parameters&  myparms,int32_t n)
  : AbstractWorker(),
    version(MCDump_run_version),
    random_01(random),
    node(n),
    parms(myparms),
    where(w),
    started(false),
    stepspersec(0.)
{
  // we really want to do measurements => create measurements object
  if( node<0||(node>=where.size()&&where.size()!=0))
    boost::throw_exception(std::logic_error("illegal node number " + boost::lexical_cast<std::string,int>(n)+" in Worker::Worker"));
  
  // TODO: create slave runs

  // TODO: replace by generic seeding scheme
  boost::minstd_rand0 gen(331);
  for (int i=0;
       i < boost::lagged_fibonacci607::long_lag*(static_cast<int>(where.size() ? static_cast<int32_t>(parms["SEED"]) : 0));
       ++i) 
    gen();
  random.seed(gen);
}


Worker::~Worker()
{// TODO: delete slave runs!!!
}


void Worker::load_worker(IDump& dump)
{

  int32_t l(dump);
  if(l!=MCDump_run)
    boost::throw_exception(std::runtime_error("dump does not contain a run"));
  int32_t u;
  dump >> u >> version;
  if(version>MCDump_run_version) {
    std::string msg = "The run on dump is version " 
        + boost::lexical_cast<std::string,int32_t>(version) + 
        + " but this program can read only up to version "
        + boost::lexical_cast<std::string,int32_t>(MCDump_run_version);
    boost::throw_exception(std::runtime_error(msg));
  }

  dump >> parms;
  std::string state;
  dump >> state;
  random = boost::lexical_cast<random_type,std::string>(state);
  if(node==0) {
    int32_t dummy;
    info.load(dump,version);
    if(version<200) 
      dump >> dummy >> dummy >> dummy;
  }
  // TODO: create slave runs
}

void Worker::save_worker(ODump& dump) const
{
  dump << int32_t(MCDump_run) << int32_t(0) << version << parms;
  dump << boost::lexical_cast<std::string,random_type>(random);
  if(node==0)
    dump << info;
  // TODO: save slave runs
 }


TaskInfo Worker::get_info() const
{
  return info;
}


void Worker::halt_worker()
{
  halt(); // user halt
  if(node==0)
    info.halt(); // store info about halting
}


void Worker::change_phase(const std::string& p)
{
  if(node==0) {
    info.halt(); // store info about halting
    info.start(p); // new phase
  }
}


// start/restart the run
void Worker::start_worker()
{
  if(node==0) {
    stepspersec=0.;
    info.start(work_phase()); // store info about starting
    }
  started=true;
  start(); // user start
  // TODO: start all slaves
  // TODO: start thread
}

		
// do some work
void Worker::run()
{
  if(started)
    dostep();
}


void Worker::write_xml(const boost::filesystem::path& , const boost::filesystem::path&) const
{
  boost::throw_exception(std::runtime_error("XML output not implemented for the worker"));
}

void Worker::load_from_file(const boost::filesystem::path& fn)
{
  IXDRFileDump dump(fn);
  load_worker(dump);
}

void Worker::save_to_file(const boost::filesystem::path& fnpath) const
{
  boost::filesystem::path bakpath=fnpath.branch_path()/(fnpath.leaf()+".bak");
  bool backup=boost::filesystem::exists(fnpath);
  {
    OXDRFileDump dump(backup ? bakpath : fnpath);
    save_worker(dump);
  } // close file
  if (backup) {
    boost::filesystem::remove(fnpath);
    boost::filesystem::rename(bakpath,fnpath);
  }
}

bool Worker::handle_message(const Process& master,int32_t tag) {
  IMPDump message;
  OMPDump dump;
  std::string name;
  alps::Parameters parms;
  switch (tag) {
    case MCMP_startRun:
      message.receive(master,MCMP_startRun);
      start_worker();
      return true;

    case MCMP_haltRun:
      message.receive(master,MCMP_haltRun);
      halt();
      return true;

    case MCMP_load_run_from_file:
      message.receive(master,MCMP_load_run_from_file);
      message >> name;
      load_from_file(boost::filesystem::path(name));
      break;
	  
    case MCMP_save_run_to_file:
      message.receive(master,MCMP_save_run_to_file);
      message >> name;
      save_to_file(boost::filesystem::path(name));
      return true;

    case MCMP_get_run_work:
      message.receive(master,MCMP_get_run_work);
      dump << work_done();
      dump.send(master,MCMP_run_work);
      return true;

    case MCMP_get_run_info:
      message.receive(master,MCMP_get_run_info);
      dump << get_info();
      dump.send(master,MCMP_run_info);
      return true;
      
    case MCMP_set_parameters:
      message.receive(master,MCMP_set_parameters);
      message >> parms;
      set_parameters(parms);
      return true;
      
    default:
      break;
  }
  return false;
}

std::string Worker::work_phase()
{
  return "";
}

void Worker::set_parameters(const alps::Parameters& p)
{
  for (Parameters::const_iterator it = p.begin(); it != p.end(); ++it) {
    if(it->key() != "SEED" && parms[it->key()] != it->value()) {
      if(!(change_parameter(it->key(), it->value()) ||
	   Worker::change_parameter(it->key(), it->value())))
	boost::throw_exception(std::runtime_error("Cannot change parameter " + it->key()));
      parms[it->key()]=it->value();
    }    
  }
}

bool Worker::change_parameter(const std::string& p, const alps::StringValue&)
{
  return p=="SEED";
}

void Worker::dostep()
{
}

double Worker::work_done() const
{
  return 0.;
}

} // namespace scheduler
} // namespace alps

