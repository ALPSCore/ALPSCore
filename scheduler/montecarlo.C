/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2002-2006 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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
#include<alps/alea.h>

#include <alps/scheduler/montecarlo.h>
#include <alps/scheduler/types.h>
#include <alps/scheduler/scheduler.h>
#include <alps/parser/xslt_path.h>
#include <alps/osiris/mpdump.h>
#include <alps/alea/detailedbinning.h>
#include <boost/filesystem/fstream.hpp>

namespace alps {
namespace scheduler {

/**
 * Returns the summary of the MCSimulation for the Observable given by the
 * name
 *
 * @params name the name of the observable
 */
ResultType MCSimulation::get_summary(const std::string name) const
{
  ResultType res;
  ObservableSet mySet = get_measurements(true);
  RealObservable* myObs = ((RealObservable*)&mySet[name]);
  res.T = parms["T"];
  res.mean = myObs->mean();
  res.error = myObs->error();
  res.count = myObs->count();
  return res;
}

/**
 * Returns the summary, the name of the Observable is specified in the job file
 */
ResultType MCSimulation::get_summary() const
{
  std::string theName;
    if (parms.defined("SUMMARY_VARIABLE"))
      theName = parms["SUMMARY_VARIABLE"];
    else
      theName = parms["ERROR_VARIABLE"];
  std::cerr << "\nMaking summary for the observable " << theName << "\n";
  if (theName.length() == 0) {
    std::cerr << "cannot find the tag ERROR_VARIABLE in the parameter set\n"
              << "so summary can be made\n";
    boost::throw_exception(std::runtime_error("no variable name to make summary after"));
  }
  return get_summary(theName);
}

// collect all measurements
ObservableSet MCSimulation::get_measurements(bool compactit) const
{
  // old measurements
  ObservableSet all_measurements;
  
  ProcessList where_master;
  int remote_runs=0;
  // add runs stored locally
  for (int i=0;i<runs.size();i++) {
    if(workerstatus[i]==RemoteRun) {
      if(!runs[i])
        boost::throw_exception(std::runtime_error( "run does not exist in MCSimulation::get_measurements"));
      where_master.push_back( Process(dynamic_cast<const RemoteWorker*>(runs[i])->process()));
      remote_runs++;
    }
    else if(runs[i]) {
      if (compactit)
        all_measurements << dynamic_cast<const MCRun*>(runs[i])->get_compacted_measurements();
      else
        all_measurements << dynamic_cast<const MCRun*>(runs[i])->get_measurements();
    }
  }
  // adding measurements from remote runs:
  if(remote_runs) {
    // broadcast request to all slaves
    OMPDump send;
    send << compactit;
    send.send(where_master,MCMP_get_measurements);
    // collect results
    for (int i=0;i<where_master.size();i++) {
      // receive dump from remote process, abort if error
      IMPDump receive(MCMP_measurements);
      ObservableSet m;
      receive >> m;
      all_measurements << m;
    }
  }
  for (ObservableSet::const_iterator it=measurements.begin();it!=measurements.end();++it)
    if (!all_measurements.has(it->first))
      all_measurements << *(it->second);
  if (compactit)
    all_measurements.compact();
  return all_measurements;
}


ObservableSet MCSimulation::get_and_remove_observable(const std::string& obsname, bool compactit) 
{
  ObservableSet obs_set;    
  ProcessList where_master;
  int remote_runs=0;
  // add runs stored locally
  for (int i=0;i<runs.size();i++) {
      if(workerstatus[i]==RemoteRun) {
	if(!runs[i])
	  boost::throw_exception(std::runtime_error( "run does not exist in MCSimulation::get_measurements"));
	where_master.push_back( Process(dynamic_cast<const RemoteWorker*>(runs[i])->process()));
	remote_runs++;
      }
      else if(runs[i]) {
	obs_set << dynamic_cast<MCRun*>(runs[i])->get_and_remove_observable(obsname, compactit);
      }
  }
  // adding measurements from remote runs:
  if(remote_runs) {
    // broadcast request to all slaves
    OMPDump send;
    send << compactit;
    send << obsname;
    send.send(where_master,MCMP_get_observable);
    // collect results
    for (int i=0;i<where_master.size();i++) {
      // receive dump from remote process, abort if error
      IMPDump receive(MCMP_observable);
      ObservableSet m;
      receive >> m;
      obs_set << m;
    }
  }
  if (measurements.has(obsname)) {
    obs_set << measurements[obsname];
    if(measurements[obsname].is_signed())
      obs_set << measurements[measurements[obsname].sign_name()];
    measurements.removeObservable(obsname);
  }
  if (compactit)
    obs_set.compact();
  return obs_set;
}


std::string MCSimulation::worker_tag() const 
{
  return "MCRUN";
}


void MCSimulation::write_xml_body(oxstream& out, const boost::filesystem::path& name) const
{
  boost::filesystem::path fn_hdf5;
  // commented out by astreich, 05/31
  // produced permament crashes.
  if(!name.empty())
    fn_hdf5=name.branch_path()/(name.leaf()+".hdf");
  // get_measurements(false).write_xml(out,name); // write non-compacted measurements
  get_measurements(false).write_xml(out,fn_hdf5); // write non-compacted measurements
  WorkerTask::write_xml_body(out,name);
}


void MCRun::print_copyright(std::ostream& out) 
{
  out << "Non-copyrighted Monte Carlo program. Please insert your own copyright statement by overwriting the print_copyright static member function of your MCRun class or the print_copyright virtual function of your factory class.\n\n";
}


MCRun::MCRun(const ProcessList& w,const Parameters&  myparms,int n)
  : Worker(w,myparms,n)
{
}


void MCRun::load_worker(IDump& dump)
{
  Worker::load_worker(dump);
  if(node==0)
    dump >> measurements;
  load(dump);
}

void MCRun::save_worker(ODump& dump) const
{
  Worker::save_worker(dump);
  if(node==0)
    dump << measurements;
  save(dump);
}

#ifdef ALPS_HAVE_HDF5
	void MCRun::serialize(hdf5::oarchive & ar) const {
		Worker::serialize(ar);
		ar << make_pvp("/simulation/realizations/0/clones/" + boost::lexical_cast<std::string>(node) + "/results", measurements);
	}
#endif

void MCRun::save(ODump&) const
{
}

void MCRun::load(IDump&)
{
}

// start/restart the run
std::string MCRun::work_phase()
{
  return is_thermalized() ? "running" : "equilibrating";
}
                
// do some work, either thermalization or measurements serrps
void MCRun::run()
{
  bool thermalized=is_thermalized();
  // do some work
  Worker::run();
  // check if it just became thermalized
  if(!thermalized&&node==0&&is_thermalized()) {
    if(node==0)
      measurements.reset(true);
    change_phase("running");
    }
}

// do one sweep
bool MCRun::is_thermalized() const
{
  // this should be implemented
  boost::throw_exception( std::logic_error("is_thermalized needs to be implemented"));
  return false;
}


void MCRun::write_xml(const boost::filesystem::path& name, const boost::filesystem::path& osirisname) const
{
  oxstream xml(name.branch_path()/(name.leaf()+".xml"));
  boost::filesystem::path fn_hdf5(name.branch_path()/(name.leaf()+".hdf"));

  xml << header("UTF-8") << stylesheet(xslt_path("ALPS.xsl"));
  xml << start_tag("SIMULATION") << xml_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
        << attribute("xsi:noNamespaceSchemaLocation","http://xml.comp-phys.org/2002/10/ALPS.xsd");
  xml << parms;
  measurements.write_xml(xml);
  xml << start_tag("MCRUN");
  if(!osirisname.empty())
    xml << start_tag("CHECKPOINT") << attribute("format","osiris")
        << attribute("file", osirisname.native_file_string())
        << end_tag("CHECKPOINT");
  xml << get_info();
  measurements.write_xml(xml,fn_hdf5);
  xml << end_tag("MCRUN") << end_tag("SIMULATION");
}


bool MCRun::handle_message(const Process& runmaster,int32_t tag)
{
  IMPDump message;
  OMPDump dump;
  ObservableSet m;
  bool compactit;
  std::string obsname;
  switch(tag) {
  case MCMP_get_measurements:
    message.receive(runmaster,MCMP_get_measurements);
    message >> compactit;
    if (compactit)
      dump << get_compacted_measurements();
    else
      dump << get_measurements();
    dump.send(runmaster,MCMP_measurements);
    return true;
  case MCMP_get_observable:
    message.receive(runmaster, MCMP_get_observable);
    message >> compactit;
    message >> obsname;
    dump << get_and_remove_observable(obsname, compactit);
    dump.send(runmaster, MCMP_observable);
    return true;
  default:
    return Worker::handle_message(runmaster,tag);
  }
}
  

ObservableSet MCRun::get_and_remove_observable(const std::string& obsname, bool compactit) 
{
  ObservableSet obs_set;
  if (measurements.has(obsname)) {
    obs_set << measurements[obsname];
    if(measurements[obsname].is_signed())
      obs_set << measurements[measurements[obsname].sign_name()];
    measurements.removeObservable(obsname);
  }
  if (compactit)
    obs_set.compact();
  return obs_set;
}


ObservableSet MCRun::get_compacted_measurements() const 
{
  ObservableSet m(get_measurements());
  m.compact();
  return m;
}


void DummyMCRun::dostep()
{
  boost::throw_exception(std::logic_error("User-level checkpointing needs to be implemented for restarting from a checkpoint\n"));
}

double DummyMCRun::work_done() const 
{
  boost::throw_exception(std::logic_error("User-level checkpointing needs to be implemented for restarting from a checkpoint\n"));
  return 0.;
}

// astreich, 06/23
ResultType DummyMCRun::get_summary() const
{
  boost::throw_exception(std::logic_error("User-level checkpointing needs to be implemented for restarting from a checkpoint\n"));
  ResultType res;
  res.count = 0;
  return res;
}

void MCSimulation::handle_tag(std::istream& infile, const XMLTag& tag) 
{
  if (tag.name!="AVERAGES")
    WorkerTask::handle_tag(infile,tag);
  else
    measurements.read_xml(infile,tag);
}

MCSimulation& MCSimulation::operator<<(const Observable& obs)
{
  addObservable(obs);
  return *this;
}

void MCSimulation::addObservable(const Observable& obs)
{
  measurements.addObservable(obs);
}

DummyMCRun::DummyMCRun()
  : MCRun(ProcessList(),Parameters(),0) 
{
}

DummyMCRun::DummyMCRun(const ProcessList& w,const Parameters& p,int n)
: MCRun(w,p,n) 
{
}

} // namespace scheduler
} // namespace alps
