/***************************************************************************
* ALPS++/scheduler library
*
* scheduler/task.C   A class to store parameters
*
* $Id$
*
* Copyright (C) 2003 by Matthias Troyer <troyer@comp-phys.org>
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

#include <alps/scheduler/task.h>
#include <alps/scheduler/types.h>
#include <alps/scheduler/scheduler.h>
#include <alps/expression.h>
#include <alps/osiris.h>
#include <alps/parser/parser.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/throw_exception.hpp>
#include <fstream>
#include <stdexcept>

namespace alps {
namespace scheduler {

Task::Task(const ProcessList& w,const boost::filesystem::path& filename)
  : AbstractTask(w),
    started(false),
    start_time(0),
    start_work(0.),
    infilename(filename)
{
}

Task::~Task()
{
  for (int i=0;i<runs.size();++i)
    if(runs[i])
      delete runs[i];
}

void Task::parse_task_file(const boost::filesystem::path& filename)
{
  boost::filesystem::ifstream infile(filename);
  
  // read outermost tag (e.g. <SIMULATION>)
  XMLTag tag=parse_tag(infile,true);
  std::string closingtag = "/"+tag.name;
  
  // scan for <PARAMETERS> and read them
  tag=parse_tag(infile,true);
  while (tag.name!="PARAMETERS" && tag.name != closingtag) {
    skip_element(infile,tag);
    tag=parse_tag(infile,true);
  }
  parms.read_xml(tag,infile);
  if (!parms.defined("SEED"))
    parms["SEED"]=0;
    
  // scan for first worker element (e.g. <MCRUN> or <REALIZATION>)
  tag=parse_tag(infile,true);
  while(true) {
    while (tag.name!=worker_tag() && tag.name != closingtag) {
      handle_tag(infile,tag);
      tag=parse_tag(infile,true);
    }
    if (tag.name==closingtag)
      break;
      
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
      boost::filesystem::path(tag.attributes["file"],boost::filesystem::native),filename.branch_path());
      // relative to XML file
    
    runfiles.push_back(files);
    workerstatus.push_back(RunOnDump);
    skip_element(infile,tag);
    while (tag.name!=worker_close) {
      skip_element(infile,tag);
      tag=parse_tag(infile,true);
    }
  }
}

void Task::handle_tag(std::istream& infile, const XMLTag& tag) 
{
  skip_element(infile,tag);
}

void Task::construct() // delayed until child class is fully constructed
{
  parse_task_file(infilename);
  runs.resize(workerstatus.size());
  ProcessList here(cpus());
  int j=-1; // count existing runs
  int in=0; // first available node
  for (int i=0;i<runs.size();i++)
  {
    j++;
    // load as many runs as possible
    if(in+cpus()<=where.size()) {// a process is available
      if(j==0&&where[in].local()) {
        // one run runs locally
#ifdef OSIRIS_TRACE
        std::cerr  << "Loading run 1 locally on " << where[0].name() << ".\n";
#endif
        std::copy(where.begin()+in,where.begin()+in+cpus(),here.begin());
        runs[0]=theScheduler->make_worker(here,parms);
	runs[0]->load_from_file(runfiles[i].in);
        theWorker = runs[0];
        workerstatus[0] = LocalRun;
        in+=cpus();
      }
      else { // load other runs onto remote nodes
#ifdef OSIRIS_TRACE
        std::cerr  << "Loading run " << j+1 << " remote on " << where[j].name() << ".\n";
#endif
        std::copy(where.begin()+in,where.begin()+in+cpus(),here.begin());
        runs[j]=new RemoteWorker(here,parms);
	runs[j]->load_from_file(runfiles[i].in);
        workerstatus[j] = RemoteRun;
        in+=cpus();
      }
    }
    else { // no node available: load information only
#ifdef OSIRIS_TRACE
      std::cerr  << "Loading information about run " << j+1 << " from file " << runfiles[i].in << ".\n";
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
#ifdef OSIRIS_TRACE
        std::cerr  << "Creating run 1 locally .\n";
#endif
        runs[0]=theScheduler->make_worker(here,parms);
        theWorker = runs[0];
        parms["SEED"] = static_cast<int32_t>(parms["SEED"])+cpus();
        in +=cpus();
        workerstatus[0] = LocalRun;
      }
      else { // other runs on remote nodes
#ifdef OSIRIS_TRACE
        std::cerr  << "Creating run " << i+1 << " remote on Host ID: " << where[i]<< ".\n";
#endif
        runs[i]=new RemoteWorker(here,parms);
        parms["SEED"] = static_cast<int32_t>(parms["SEED"])+cpus();
        in +=cpus();
        workerstatus[i] = RemoteRun;
      }
    }
  }
  for (int i=0;i<runs.size();++i)
    runs[i]->set_parameters(parms);
}
        
	
// start all runs which are active
void Task::start()
{
  if(!started) {
    started=true;
    for (int i=0; i<runs.size();i++)
      if(runs[i] && workerstatus[i] > RunNotExisting && workerstatus[i] < RunOnDump)
        runs[i]->start_worker();
  }
}


// start an extra run on a new node
void Task::add_process(const Process& p)
{
  ProcessList here(1);
  here[0]=p;

  int i;
  // look for empty entry
  for (i=0;i<where.size() && where[i].valid();i++)
    {}   
  if(i==where.size())
    where.resize(i+1);
  where[i] = p;
  
  int j;
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
#ifdef OSIRIS_TRACE
    std::cerr  << "Creating additional run " << j+1 << " remote on Host: " << p.name() << ".\n";
#endif
    runs[j]=new RemoteWorker(here,parms);
    parms["SEED"] = static_cast<int32_t>(parms["SEED"])+cpus();
    workerstatus[j] = RemoteRun;
    if(started)
      runs[j]->start_worker();
  }
  else {// continue old run
#ifdef OSIRIS_TRACE
    std::cerr  << "Loading additional run " << j << " remote on Host: " << p.name() << ".\n";
#endif
    runs[j]=new RemoteWorker(here,parms);
    runs[j]->load_from_file(runfiles[j].in);
    workerstatus[j] = RemoteRun;
  }
}


// remove one run : hope that a checkpoint was created before!!!
void Task::delete_process(const Process& p)
{
  ProcessList::iterator found = std::find(where.begin(),where.end(),p);
  if( found==where.end())
    return;
  // delete process from list
  *found = Process();
  ProcessList nowhere;
  
  std::cerr << "Deleting run on " << p.name() << ".\n";
  // change status of run, eventually reload it from dump
  int found_run=(found-where.begin()) / cpus();
  if(workerstatus[found_run] == LocalRun || workerstatus[found_run] == RemoteRun)
    if(!runfiles[found_run].in.empty()) { // reload info from file
      workerstatus[found_run] = RunOnDump;
      if (runs[found_run])
        delete runs[found_run];
      runs[found_run]=theScheduler->make_worker(parms);
      runs[found_run]->load_from_file(runfiles[found_run].in);
    }
    else {
      workerstatus[found_run] = RunNotExisting;
      runs[found_run]=0;
    }
}


// is it finished???
bool Task::finished(double& more_time) const
{
  static double old_work;

  // get work estimate
  double w = work();
  if(w<=0.)
    return true;

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
  return 0;
}


// do some work on the local run
void Task::run()
{
  if(started&&theWorker)
    dynamic_cast<Worker&>(*theWorker).run();
}


// halt all active runs
void Task::halt()
{
  if(started) {
    started=false;
    for(int i=0;i<runs.size();i++)
      if(runs[i] && workerstatus[i] > RunNotExisting && workerstatus[i] < RunOnDump)
        runs[i]->halt_worker();
  }
}

double Task::work_done()  const
{
  double w=0.;
  ProcessList where_master;

  // add runs stored locally
  if(runs.size()) {
    for (int i=0;i<runs.size();i++) {
      if(workerstatus[i]==RemoteRun) {
	if(!runs[i])
	    boost::throw_exception(std::runtime_error( "run does not exist in Task::get_measurements"));
	where_master.push_back( Process(dynamic_cast<RemoteWorker&>(*runs[i]).process()));
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
    for (int i=0;i<where_master.size();i++) {
      // receive dump from remote process, abort if error
      IMPDump receive(MCMP_run_work);
      w += double(receive);
    }
  }
  return w;
}


double Task::work() const
{
  return (parms.defined("WORK_FACTOR") ? evaluate(parms["WORK_FACTOR"], parms) : 1. )
         *(1.-work_done());
}

// checkpoint: save into a file
void Task::checkpoint(const boost::filesystem::path& fn) const
{
  boost::filesystem::path dir=fn.branch_path();
  bool make_backup = boost::filesystem::exists(fn);
  boost::filesystem::path filename = (make_backup ? dir/(fn.leaf()+".bak") : fn);
  {
  boost::filesystem::ofstream out (filename);
  
  write_xml_header(out);
  parms.write_xml(out);
  write_xml_body(out,fn);
  for (int i=0;i<runs.size();++i) {
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
	  name.erase(name.size()-4,4);
          name+= ".run" + boost::lexical_cast<std::string,int>(j+1);
	  for (int k=0;k<runfiles.size();++k)
	  if(runfiles[k].out.leaf()==name) 
	    found=true;
	  j++;
	} while (found);
	runfiles[i].out = boost::filesystem::path(name);
      }
      if(workerstatus[i] == LocalRun || workerstatus[i] == RemoteRun)
	runs[i]->save_to_file(boost::filesystem::complete(runfiles[i].out,dir));
      else if (workerstatus[i] == RunOnDump) {
        if(boost::filesystem::complete(runfiles[i].out,dir).string()!=runfiles[i].in.string()) {
	  boost::filesystem::remove(boost::filesystem::complete(runfiles[i].out,dir));
	  boost::filesystem::copy_file(boost::filesystem::complete(runfiles[i].in,dir),boost::filesystem::complete(runfiles[i].out,dir));
	}
      }
      else 
	boost::throw_exception(std::logic_error("incorrect status of run"));
      out << "  <" << worker_tag() << ">\n";
      out << runs[i]->get_info();
      out << "    <CHECKPOINT file=\"" << runfiles[i].out.native_file_string() << "\"/>\n";
      out << "  </" << worker_tag() << ">\n";
      runfiles[i].in=boost::filesystem::complete(runfiles[i].out,dir);
    }
  }
  write_xml_trailer(out);
  } // close file
  if(make_backup) {
    boost::filesystem::remove(fn);
    boost::filesystem::rename(filename,fn);
  }
}

} // namespace scheduler
} // namespace alps
