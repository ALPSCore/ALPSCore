/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2006 by Matthias Troyer <troyer@comp-phys.org>,
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

#include <alps/scheduler/task.h>
#include <alps/scheduler/types.h>
#include <alps/scheduler/scheduler.h>
#include <alps/expression.h>
#include <alps/parser/parser.h>
#include <alps/osiris/mpdump.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/throw_exception.hpp>
#include <fstream>
#include <stdexcept>

#define ALPS_TRACE

namespace alps {
namespace scheduler {

WorkerTask::WorkerTask(const ProcessList& w,const boost::filesystem::path& filename)
  : Task(w,filename),
    start_time(0),
    start_work(0.)
{
}

WorkerTask::WorkerTask(const ProcessList& w,const Parameters& p)
  : Task(w,p),
    start_time(0),
    start_work(0.)
{
}

WorkerTask::~WorkerTask()
{
  for (unsigned int i=0;i<runs.size();++i)
    if(runs[i])
      delete runs[i];
}

#ifdef ALPS_HAVE_HDF5
	void WorkerTask::serialize(hdf5::iarchive & ar) {
		std::vector<std::string> list = ar.list_children("/checkpoint");
		for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
			std::string filename;
			ar >> make_pvp("/checkpoint/" + *it, filename);
			CheckpointFiles files;
			files.in = boost::filesystem::complete(boost::filesystem::path(filename, boost::filesystem::native), infilename.branch_path());
			runfiles.push_back(files);
			workerstatus.push_back(RunOnDump);
		}
	}
#endif;

void WorkerTask::handle_tag(std::istream& infile, const XMLTag& intag) 
{
  if (intag.name!=worker_tag()) {
    Task::handle_tag(infile,intag);
    return;
  }
  
  XMLTag tag=intag;
  // scan for <CHECKPOINT> tag
  if (tag.type==XMLTag::SINGLE)
    boost::throw_exception(std::runtime_error("<CHECKPOINT> element missing in task file"));
  std::string worker_close ="/"+worker_tag();
  tag=parse_tag(infile,true);
  while (tag.name!="CHECKPOINT") {
    if(tag.name==worker_close)
      boost::throw_exception(std::runtime_error("<CHECKPOINT> element missing in task file"));
    skip_element(infile,tag);
    tag=parse_tag(infile,true);
  }
    
  // read <CHECKPOINT> tag
  if (tag.attributes["file"]=="")
    boost::throw_exception(std::runtime_error("file attribute missing in <CHECKPOINT> element in task file"));
  CheckpointFiles files; 
  files.in=boost::filesystem::complete(
  boost::filesystem::path(tag.attributes["file"],boost::filesystem::native),infilename.branch_path());
  // relative to XML file
    
  runfiles.push_back(files);
  workerstatus.push_back(RunOnDump);
  skip_element(infile,tag);
  while (tag.name!=worker_close) {
    skip_element(infile,tag);
    tag=parse_tag(infile,true);
  }
}

void WorkerTask::construct() // delayed until child class is fully constructed
{
  Task::construct();
  runs.resize(workerstatus.size());
  ProcessList here(cpus());
  int j=-1; // count existing runs
  int in=0; // first available node
  for (unsigned int i=0;i<runs.size();i++) {
    j++;
    // load as many runs as possible
    if(in+cpus()<=where.size()) {// a process is available
      if(j==0&&where[in].local()) {
        // one run runs locally
#ifdef ALPS_TRACE
        std::cerr  << "Loading run 1 locally on " << where[0] << "\n";
#endif
        std::copy(where.begin()+in,where.begin()+in+cpus(),here.begin());
        runs[0]=theScheduler->make_worker(here,parms);
        runs[0]->load_from_file(runfiles[i].in);
        theWorker = runs[0];
        workerstatus[0] = LocalRun;
        in+=cpus();
      }
      else { // load other runs onto remote nodes
#ifdef ALPS_TRACE
        std::cerr  << "Loading run " << j+1 << " remote on " << where[j] << "\n";
#endif
        std::copy(where.begin()+in,where.begin()+in+cpus(),here.begin());
        runs[j]=new RemoteWorker(here,parms);
        runs[j]->load_from_file(runfiles[i].in);
        workerstatus[j] = RemoteRun;
        in+=cpus();
      }
    }
    else { // no node available: load information only
#ifdef ALPS_TRACE
      std::cerr  << "Loading information about run " << j+1 << " from file "
                 << runfiles[i].in.string() << "\n";
#endif
      runs[j]=theScheduler->make_worker(parms);
      runs[j]->load_from_file(runfiles[i].in);
      workerstatus[j] = RunOnDump;
    }
  }

  if(in+cpus()<=where.size()) { // more nodes than runs dumped: create extra runs
    runs.resize(where.size()/cpus());
    workerstatus.resize(where.size()/cpus());
    runfiles.resize(where.size()/cpus());
    for(int i=j+1;in+cpus()<=where.size();i++)
    {
      std::copy(where.begin()+in,where.begin()+in+cpus(),here.begin());
      if(in==0&&here[0].local()) { // one on the local node
        runs[0]=theScheduler->make_worker(here,parms);
        theWorker = runs[0];
        parms["SEED"] = static_cast<int32_t>(parms["SEED"])+cpus();
        in +=cpus();
        workerstatus[0] = LocalRun;
#ifdef ALPS_TRACE
        std::cerr  << "Created run 1 locally\n";
#endif
      }
      else { // other runs on remote nodes
        runs[i]=new RemoteWorker(here,parms);
        parms["SEED"] = static_cast<int32_t>(parms["SEED"])+cpus();
        in +=cpus();
        workerstatus[i] = RemoteRun;
#ifdef ALPS_TRACE
        std::cerr  << "Created run " << i+1 << " remote on Host ID: "
                   << where[i]<< "\n";
#endif
      }
    }
  }
  for (unsigned int i=0;i<runs.size();++i) {
    runs[i]->set_parameters(parms);
  }
}
        
// start all runs which are active
void WorkerTask::start()
{
  if(!started()) {
    Task::start();
    for (unsigned int i=0; i<runs.size();i++)
      if(runs[i] && workerstatus[i] > RunNotExisting && workerstatus[i] < RunOnDump) {
        runs[i]->start_worker();
      }
  }
}


// start an extra run on a new node
void WorkerTask::add_process(const Process& p)
{
  ProcessList here(1);
  here[0]=p;

  unsigned int i;
  // look for empty entry
  for ( i=0;i<where.size() && where[i].valid();i++)
    {}   
  if(i==where.size())
    where.resize(i+1);
  where[i] = p;
  
  unsigned int j;
  // look for run to start on this process
  for (j=0; j<runs.size() && runs[j] && workerstatus[j] != RunNotExisting 
                              && workerstatus[j] != RunOnDump ; j++)
    {}
    
  if(i != j)
    boost::throw_exception(std::logic_error( "In Task::add_process: # running runs != # running processes"));
  
  if(j==runs.size() || workerstatus[j] != RunOnDump) { // start new run
    runs.resize(j+1);
    workerstatus.resize(j+1);
    runfiles.resize(j+1);
#ifdef ALPS_TRACE
    std::cerr  << "Creating additional run " << j+1 << " remote on Host: " << p << "\n";
#endif
    runs[j]=new RemoteWorker(here,parms);
    parms["SEED"] = static_cast<int32_t>(parms["SEED"])+cpus();
    workerstatus[j] = RemoteRun;
    if(started())
      runs[j]->start_worker();
  }
  else {// continue old run
#ifdef ALPS_TRACE
    std::cerr  << "Loading additional run " << j << " remote on Host: " << p << "\n";
#endif
    runs[j]=new RemoteWorker(here,parms);
    runs[j]->load_from_file(runfiles[j].in);
    workerstatus[j] = RemoteRun;
  }
}



// is it finished???
bool WorkerTask::finished(double& more_time, double& percentage) const
{
  if (finished_)
    return true;

  // get work estimate
  double w = work();
  if(w<=0.)
    return true;

  percentage = 1.-w;
  if (percentage < 0.) 
    percentage=0.;
  else if (percentage>1.)
    percentage=1.;
  // estinate remaining time
  if(more_time<0)
    start_time=0; // new estimate

  if(start_time==0) { // initialize timing
    start_time=time(0);
    start_work=w;
    old_work=w;
  }
  else if(start_work==old_work) {
    start_time=time(0);
    if(w!=old_work) {
      start_work=w;
      old_work=-1;
    }
  }
  else if(start_work>w) { 
    // estimate remaining time
    // propose to run 1/4 of that time
    time_t now = time(0);
    more_time = 0.25*w*(now-start_time)/(start_work-w);
  }
  return false;
}

// do some work on the local run
void WorkerTask::dostep()
{
  if(theWorker)
    dynamic_cast<Worker&>(*theWorker).run();
}

// halt all active runs
void WorkerTask::halt()
{
  if(started()) {
    Task::halt();
    for(unsigned int i=0;i<runs.size();i++)
      if(runs[i] && workerstatus[i] > RunNotExisting && workerstatus[i] < RunOnDump)
        runs[i]->halt_worker();
  }
}

ResultType WorkerTask::get_summary() const
{
  ResultType res;
  res.mean=0.;
  res.error=0.;
  res.count=0.;
  
  ProcessList where_master;

  // add runs stored locally
  if (runs.size()) {
    for (unsigned int i=0; i<runs.size(); i++) {
      if (workerstatus[i]==RemoteRun) {
        if (!runs[i])
          boost::throw_exception(std::runtime_error("Run does not exist in Task::get_measurements"));
        where_master.push_back(dynamic_cast<RemoteWorker*>(runs[i])->process());
      }
      else if (runs[i])
        res += runs[i]->get_summary();
    }
  }

  if (where_master.size()) {
    // broadcast request to all slaves
    OMPDump send;
    send.send(where_master,MCMP_get_summary);
    
    // collect results
    for (unsigned int i=0; i<where_master.size(); i++) {
      // receive dump
      IMPDump receive(MCMP_summary);
      ResultType s_res;
      receive >> s_res.T >> s_res.mean >> s_res.error >> s_res.count;
      res += s_res;
    }
  }
  return res;
}

double WorkerTask::work_done()  const
{
  double w=0.;
  ProcessList where_master;
  
  // add runs stored locally
  if(runs.size()) {
    for (unsigned int i=0;i<runs.size();i++) {
      if(workerstatus[i]==RemoteRun) {
         if(!runs[i])
            boost::throw_exception(std::runtime_error( "run does not exist in Task::get_measurements"));
        //where_master.push_back( Process(dynamic_cast<RemoteWorker&>(*runs[i]).process()));
        where_master.push_back(dynamic_cast<RemoteWorker*>(runs[i])->process());
      }
      else if(runs[i])
        w += runs[i]->work_done();
    }
  }

  // adding measurements from remote runs:
  if(where_master.size()) {
    // broadcast request to all slaves
    OMPDump send;
    send.send(where_master,MCMP_get_run_work);
      
    // collect results
    for (unsigned int i=0;i<where_master.size();i++) {
      // receive dump from remote process, abort if error
      IMPDump receive(MCMP_run_work);
      w += double(receive);
    }
  }
  return w;
}

double WorkerTask::work() const
{
  if (finished_)
    return 0.;
  return (parms.defined("WORK_FACTOR") ? alps::evaluate<double>(parms["WORK_FACTOR"], parms) : 1. )
         *(1.-work_done());
}

// checkpoint: save into a file
void WorkerTask::write_xml_body(alps::oxstream& out, const boost::filesystem::path& fn) const
{
  boost::filesystem::path dir=fn.branch_path();
#ifdef ALPS_HAVE_HDF5
	std::string task_path = fn.file_string().substr(0, fn.file_string().find_last_of('.')) + ".h5";
	std::string task_backup = fn.file_string().substr(0, fn.file_string().find_last_of('.')) + ".bak.h5";
	bool task_exists = boost::filesystem::exists(task_path);
	hdf5::oarchive task_ar(task_exists ? task_backup : task_path);
	task_ar
		<< make_pvp("/parameters", parms) 
	;
#endif
  for (unsigned int i=0;i<runs.size();++i) {
    if(workerstatus[i] == RunNotExisting) {
      if(runs[i])
        boost::throw_exception(std::logic_error("run exists but marked as non-existing"));
    }
    else if(runs[i]==0)
      boost::throw_exception(std::logic_error("run does not exist but marked as existing"));
    else {
      if (!runfiles[i].out.empty())
        runfiles[i].in=boost::filesystem::complete(runfiles[i].out,dir);
      else {
        // search file name
        int j=0;
        bool found=false;
        std::string name;
        do {
          found = false;
          name =fn.leaf();
          name = name.substr(0, name.find_last_of('.'));
          name+= ".run" + boost::lexical_cast<std::string,int>(j+1);
          for (unsigned int k=0;k<runfiles.size();++k)
          if(runfiles[k].out.leaf()==name) 
            found=true;
          j++;
        } while (found);
        runfiles[i].out = boost::filesystem::path(name);
      }
      if(workerstatus[i] == LocalRun || workerstatus[i] == RemoteRun) {
        runs[i]->save_to_file(boost::filesystem::complete(runfiles[i].out,dir));
#ifdef ALPS_HAVE_HDF5
		{
			std::string worker_path = boost::filesystem::complete(runfiles[i].out, dir).file_string() + ".h5";
			std::string worker_backup = boost::filesystem::complete(runfiles[i].out, dir).file_string() + ".bak.h5";
			bool worker_exists = boost::filesystem::exists(worker_path);
			{
				hdf5::oarchive worker_ar(worker_exists ? worker_backup : worker_path);
				worker_ar << make_pvp("/", runs[i]);
			}
			if (worker_exists) {
				boost::filesystem::remove(worker_path);
				boost::filesystem::rename(worker_backup, worker_path);
			}
		}
#endif
      } else if (workerstatus[i] == RunOnDump) {
        if(boost::filesystem::complete(runfiles[i].out,dir).string()!=runfiles[i].in.string()) {
          boost::filesystem::remove(boost::filesystem::complete(runfiles[i].out,dir));
          boost::filesystem::copy_file(boost::filesystem::complete(runfiles[i].in,dir),boost::filesystem::complete(runfiles[i].out,dir));
        }
      }
      else 
        boost::throw_exception(std::logic_error("incorrect status of run"));
      out << alps::start_tag(worker_tag());
      out << runs[i]->get_info();
      out << alps::start_tag("CHECKPOINT") << alps::attribute("format","osiris")
          << alps::attribute("file",runfiles[i].out.native_file_string());
      out << alps::end_tag("CHECKPOINT") << alps::end_tag(worker_tag());
      runfiles[i].in=boost::filesystem::complete(runfiles[i].out,dir);
#ifdef ALPS_HAVE_HDF5
		task_ar
			<< make_pvp("/logs/" + boost::lexical_cast<std::string>(i), runs[i]->get_info())
			<< make_pvp("/checkpoint/" + boost::lexical_cast<std::string>(i), runfiles[i].out.file_string())
//			<< make_pvp("/checkpoint/" + boost::lexical_cast<std::string>(i), runfiles[i].out.file_string() + ".h5")
		;
#endif
    }
  }
#ifdef ALPS_HAVE_HDF5
	if (task_exists) {
		boost::filesystem::remove(task_path);
		boost::filesystem::rename(task_backup, task_path);
	}
#endif
}

} // namespace scheduler
} // namespace alps
 