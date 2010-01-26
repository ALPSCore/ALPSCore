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

    engine_ptr(rng_factory.create(myparms.value_or_default("RNG","mt19937"))),
    random(*engine_ptr, boost::uniform_real<>()),
    random_01(*engine_ptr, boost::uniform_real<>()),

    node(n),
    parms(myparms),
    where(w),
    started(false)
{
  if( node<0||(node>=where.size()&&where.size()!=0))
    boost::throw_exception(std::logic_error("illegal node number " + boost::lexical_cast<std::string,int>(n)+" in Worker::Worker"));
  
  // TODO: create slave runs

  if (where.size()) engine_ptr->seed(parms["SEED"]);

  Disorder::seed(parms.value_or_default("DISORDERSEED",0));
}

Worker::Worker(const alps::Parameters&  myparms,int32_t n)
  : AbstractWorker(),
    version(MCDump_worker_version),

    engine_ptr(rng_factory.create(myparms.value_or_default("RNG","mt19937"))),
    random(*engine_ptr, boost::uniform_real<>()),
    random_01(*engine_ptr, boost::uniform_real<>()),

    node(n),
    parms(myparms),
    where(1),
    started(false)
{
  if( node<0||(node>=where.size()&&where.size()!=0))
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
  // TODO: load slave runs
}

void Worker::save_worker(ODump& dump) const
{
  dump << int32_t(MCDump_run) << int32_t(0) << int32_t(MCDump_worker_version) << parms;
  std::ostringstream rngstream;

  rngstream << *engine_ptr;

  dump << rngstream.str();
  if(node==0)
    dump << info;
  // TODO: save slave runs
 }
 
#ifdef ALPS_HAVE_HDF5
	void Worker::serialize(hdf5::iarchive & ar) {
		int run;
		std::string state;
		ar 
			>> make_pvp("/run", run) 
			>> make_pvp("/version", version) 
			>> make_pvp("/parameters", parms) 
			>> make_pvp("/engine_ptr", state)
		;
		if(run != MCDump_run)
			boost::throw_exception(std::runtime_error("dump does not contain a run"));
		if(version > MCDump_worker_version) {
			std::string msg = "The run on dump is version " 
				+ boost::lexical_cast<std::string>(version) + 
				+ " but this program can read only up to version "
				+ boost::lexical_cast<std::string>(MCDump_worker_version);
			throw std::runtime_error(msg);
		}
		std::stringstream rngstream(state);
		engine_ptr->read(rngstream);
		if(node == 0)
			ar >> make_pvp("/info", info);
		Disorder::seed(parms.value_or_default("DISORDERSEED",0));
	}
	void Worker::serialize(hdf5::oarchive & ar) const {
		std::ostringstream rngstream;
		rngstream << *engine_ptr;
		ar 
			<< make_pvp("/run", int(MCDump_run)) 
			<< make_pvp("/version", int(MCDump_worker_version)) 
			<< make_pvp("/parameters", parms) 
			<< make_pvp("/engine_ptr", rngstream.str())
		;
			if(node == 0)
				ar << make_pvp("/info", info);
	}
#endif

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


void Worker::write_xml(const boost::filesystem::path& , const boost::filesystem::path&) const
{
  boost::throw_exception(std::runtime_error("XML output not implemented for the worker"));
}

void Worker::load_from_file(const boost::filesystem::path& fn)
{
#ifdef ALPS_HAVE_HDF5
	if (fn.file_string().substr(fn.file_string().size() - 3) == ".h5") {
		hdf5::iarchive ar(fn.file_string());
		ar >> make_pvp("/", this);
	} else
#endif
	{
  IXDRFileDump dump(fn);
  load_worker(dump);
}
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

    case MCMP_get_summary:
      // return the summary of this task to the master
      message.receive(master,MCMP_get_summary);
      res = get_summary();
      dump << res.T << res.mean << res.error << res.count;
      dump.send(master,MCMP_summary);
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
// TODO: Lukas
          // check why the values are not the same
          // print a warning and print the values to check instead of throwing an exception, but accepty the change for now
          // maybe we need to also test whether the partially or fully evaluated values differ
          std::cerr << "parameters do not match: " << it->key() << ", value: " << parms[it->key()] << ", value2: " << it->value() << std::endl;
//        boost::throw_exception(std::runtime_error("Cannot change parameter " + it->key()));
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
  ResultType res;
  std::cerr << "\nWorker:;get_summary() called - this should not happen!!\n";
  return res;
}
} // namespace scheduler
} // namespace alps

