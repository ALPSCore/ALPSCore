/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2005 by Matthias Troyer <troyer@comp-phys.org>
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

#include <alps/random.h>
#include <alps/scheduler/scheduler.h>
#include <alps/expression.h>
#include <alps/osiris/std/string.h>
#include <alps/osiris/comm.h>
#include <alps/osiris/xdrdump.h>
#include <alps/osiris/mpdump.h>

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
    version(MCDump_worker_version),
    node(n),
    parms(myparms),
    where(w),
    engine_ptr(rng_factory.create(rng_name())),
    random(*engine_ptr, boost::uniform_real<>()),
    random_01(*engine_ptr, boost::uniform_real<>()),
    started(false)
{
  if( node<0||(node>=(int32_t)where.size()&&where.size()!=0))
    boost::throw_exception(std::logic_error("illegal node number " + boost::lexical_cast<std::string,int>(n)+" in Worker::Worker"));
  
  // TODO: create slave runs

  if (where.size()) engine_ptr->seed(parms["SEED"]);

  Disorder::seed(parms.value_or_default("DISORDERSEED",0));
}

Worker::Worker(const alps::Parameters&  myparms,int32_t n)
  : AbstractWorker(),
    version(MCDump_worker_version),
    node(n),
    parms(myparms),
    where(1),
    engine_ptr(rng_factory.create(rng_name())),
    random(*engine_ptr, boost::uniform_real<>()),
    random_01(*engine_ptr, boost::uniform_real<>()),
    started(false)
{
  if( node<0||(node>=(int32_t)where.size()&&where.size()!=0))
    boost::throw_exception(std::logic_error("illegal node number " + boost::lexical_cast<std::string,int>(n)+" in Worker::Worker"));
  
  // TODO: create slave runs

  if (where.size()) engine_ptr->seed(parms["SEED"]);

  Disorder::seed(parms.value_or_default("DISORDERSEED",0));
}


Worker::~Worker()
{
  // TODO: delete slave runs!!!
}


void Worker::load_worker(IDump& dump)
{
  int32_t l(dump);
  if(l!=MCDump_run)
    boost::throw_exception(std::runtime_error("dump does not contain a run"));
  int32_t u;
  dump >> u >> version;
  dump.set_version(version);

  if(version>MCDump_worker_version) {
    std::string msg = "The run on dump is version " 
        + boost::lexical_cast<std::string,int32_t>(version) + 
        + " but this program can read only up to version "
        + boost::lexical_cast<std::string,int32_t>(MCDump_worker_version);
    boost::throw_exception(std::runtime_error(msg));
  }

  if (version < 400) {
    dump >> parms;
    std::string state;
    dump >> state;
    std::stringstream rngstream(state);

    if (version < 304 && !parms.defined("RNG"))
      std::clog << "Re-seeding the random number generator since its type has changed from the old version. Please define RNG to the old value of \"lagged_fibonacci607\" to continue with the old generator." << std::endl;
    else
      engine_ptr->read(rngstream);

    if(node==0) {
      int32_t dummy;
      info.load(dump,version);
      if(version<200) 
        dump >> dummy >> dummy >> dummy;
    }
    Disorder::seed(parms.value_or_default("DISORDERSEED",0));
  }
}

void Worker::save_worker(ODump& dump) const
{
  dump << int32_t(MCDump_run) << int32_t(0) << int32_t(MCDump_worker_version);
  if (MCDump_worker_version < 400) {
    dump << parms;
    std::ostringstream rngstream;

    rngstream << *engine_ptr;

    dump << rngstream.str();
    if(node==0)
      dump << info;
  }
 }
 
void Worker::load(hdf5::archive & ar) 
{
  std::string state;
  std::string rngname;
  ar  >> make_pvp("/parameters", parms)
      >> make_pvp("/rng", state)
      >> make_pvp("/rng/@name", rngname);

  std::stringstream rngstream(state);
  if (rngname != rng_name())
    boost::throw_exception(std::runtime_error("Created RNG " + rng_name() + " but attenprint to load " + rngname));
  engine_ptr->read(rngstream);
  if(node == 0)
      ar >> make_pvp("/log/alps", info);
  Disorder::seed(parms.value_or_default("DISORDERSEED",0));
}

void Worker::save(hdf5::archive & ar) const 
{
  std::ostringstream rngstream;
  rngstream << *engine_ptr;
  ar << make_pvp("/parameters", parms) 
      << make_pvp("/rng", rngstream.str())
      << make_pvp("/rng/@name", rng_name());
  if(node == 0)
      ar << make_pvp("/log/alps", info);
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
  if(node==0) 
    info.start(work_phase()); // store info about starting
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


void Worker::write_xml(const boost::filesystem::path&) const
{
  boost::throw_exception(std::runtime_error("XML output not implemented for the worker"));
}

void Worker::load_from_file(const boost::filesystem::path& fn,const boost::filesystem::path& hdf5path)
{
#ifdef ALPS_HAVE_HDF5
  if (boost::filesystem::exists(hdf5path)) {
      hdf5::archive ar(hdf5path.file_string());
      ar >> make_pvp("/", *this);
  }
#endif
  IXDRFileDump dump(fn);
  load_worker(dump);
}

void Worker::save_to_file(const boost::filesystem::path& fnpath, const boost::filesystem::path& hdf5path) const
{
  boost::filesystem::path bakpath=fnpath.branch_path()/(fnpath.leaf()+".bak");
  bool backup=boost::filesystem::exists(fnpath);
  
#ifdef ALPS_HAVE_HDF5
  boost::filesystem::path hdf5bakpath =  fnpath.branch_path()/(hdf5path.leaf()+".bak");
  backup =  backup || boost::filesystem::exists(fnpath);
  {
    boost::filesystem::path p = backup ? hdf5bakpath : hdf5path;
    if (boost::filesystem::exists(p))
      boost::filesystem::remove(p);
    hdf5::archive worker_ar(p.string(), hdf5::archive::WRITE);
    worker_ar << make_pvp("/", *this);
  } // close file
  if (backup) {
    if (boost::filesystem::exists(hdf5path))
      boost::filesystem::remove(hdf5path);
    boost::filesystem::rename(hdf5bakpath,hdf5path);
  }
#endif

  {
    OXDRFileDump dump(backup ? bakpath : fnpath);
    save_worker(dump);
  } // close file
  if (backup) {
    if (boost::filesystem::exists(fnpath))
      boost::filesystem::remove(fnpath);
    boost::filesystem::rename(bakpath,fnpath);
  }
}

bool Worker::handle_message(const Process& master,int32_t tag) {
  IMPDump message;
  OMPDump dump;
  std::string name1, name2;
  alps::Parameters parms;
  ResultType res;
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
      message >> name1 >> name2;
      load_from_file(boost::filesystem::path(name1),boost::filesystem::path(name2));
      break;
          
    case MCMP_save_run_to_file:
      message.receive(master,MCMP_save_run_to_file);
      message >> file_name1 >> file_name2;
      save_to_file(boost::filesystem::path(file_name1),boost::filesystem::path(file_name2));
      return true;

    case MCMP_get_run_work:
      message.receive(master,MCMP_get_run_work);
      dump << work_done();
      dump.send(master,MCMP_run_work);
      return true;

    case MCMP_get_summary:
      // return the summary of this task to the master
      message.receive(master,MCMP_get_summary);
      res = get_summary();
      dump << res.T << res.mean << res.error << res.count;
      dump.send(master,MCMP_summary);
      return true;

    case MCMP_get_run_info:
      message.receive(master,MCMP_get_run_info);
      dump << get_info() << file_name1 << file_name2;
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
    if(it->key() != "SEED" && 
      !same_values(simplify_value(parms[it->key()],parms),simplify_value(it->value(),p),1e-6)) {
      if(!(change_parameter(it->key(), it->value()) ||
          Worker::change_parameter(it->key(), it->value()))) {
        std::cerr << "parameters do not match: " << it->key() << ", value: " 
        << parms[it->key()] << " [= " << simplify_value(parms[it->key()],parms)
        << "], value2: " 
        << it->value() << " [= " << simplify_value(it->value(),p) << "]" << std::endl;
        boost::throw_exception(std::runtime_error("Cannot change parameter " + it->key()));
      }
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

ResultType Worker::get_summary() const
{
  std::cerr << "\nWorker:;get_summary() called - this should not happen!!\n";
  return ResultType();
}
} // namespace scheduler
} // namespace alps

