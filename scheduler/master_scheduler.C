/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@comp-phys.org>
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

#include <alps/scheduler/scheduler.h>
#include <alps/scheduler/signal.hpp>
#include <alps/scheduler/types.h>
#include <alps/osiris.h>
#include <alps/parser/parser.h>
#include <alps/parser/xmlstream.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/smart_ptr.hpp>
#include <fstream>

namespace alps {

namespace scheduler {

// astreich, 07/05
MasterScheduler::MasterScheduler(const NoJobfileOptions& opt,const Factory& p)
  : Scheduler(opt,p),
    min_check_time(opt.min_check_time),
    max_check_time(opt.max_check_time),
    checkpoint_time(opt.checkpoint_time),
    min_cpus(opt.min_cpus),
    max_cpus(opt.max_cpus),
    time_limit(opt.time_limit)
{
  // register all processes and hosts with the MP signal handler
  if(opt.programname.size()!=0)
    processes=start_all_processes(programname);
}


MasterScheduler::MasterScheduler(const Options& opt,const Factory& p)
  : Scheduler(opt,p),
    min_check_time(opt.min_check_time),
    max_check_time(opt.max_check_time),
    checkpoint_time(opt.checkpoint_time),
    min_cpus(opt.min_cpus),
    max_cpus(opt.max_cpus),
    time_limit(opt.time_limit)
{
  // register all processes and hosts with the MP signal handler
  if(opt.programname.size()!=0)
    processes=start_all_processes(programname);

  // the rest of the initialisation is done by the set_new_jobfile - function
  // this function can also be used to re-start the scheduler with a new
  // job file.
  set_new_jobfile(opt.jobfilename);
}

/**
 * Registers a new job file and updates everything that is related to the job
 * files - namely the task files are parsed and the tasks are created
 * accordingly.
 * This function is used for the fitting and allows to start another simulation
 * based on newly created files. Deleting and re-creating the schedulers yields
 * to problems with syncronisation and integrity of the message space.
 *
 * @params jobfilename The name of the new job file
 */
// astreich, 06/30
void MasterScheduler::set_new_jobfile(boost::filesystem::path jobfilename)
{
  // clean the 'traces' of the previous simulation
  taskfiles.clear();
  tasks.clear();
  taskstatus.clear();
  sim_results.clear();

  outfilepath = jobfilename;
  infilepath = jobfilename;
  
  infilepath=boost::filesystem::complete(infilepath);
  outfilepath=boost::filesystem::complete(outfilepath);

  if (!jobfilename.empty())
    parse_job_file(infilepath);
      
  tasks.resize(taskfiles.size());
  sim_results.resize(taskfiles.size());

  std::cerr << "parsing task files ... \n";
  for (int i=0; i<taskfiles.size(); i++) {
#ifndef BOOST_NO_EXCEPTIONS
    try {
#endif
      tasks[i]=make_task(taskfiles[i].in);
      if (tasks[i] && 
          taskstatus[i]!= TaskFinished && tasks[i]->finished_notime())  {
        tasks[i]->start();
        std::cout << "Task " << i+1 << " is actually finished.\n";
        finish_task(i);
      }

#ifndef BOOST_NO_EXCEPTIONS
    }
    catch (const std::runtime_error& err) // file does not exist
    {
      std::cerr << err.what() << "\n";
      std::cerr  << "Cannot open simulation file " << taskfiles[i].in.string() 
                 << ".\n";
      tasks[i]=0;
      taskstatus[i] = TaskNotExisting;
    }
#endif
  }
}

void MasterScheduler::parse_job_file(const boost::filesystem::path& filename)
{
  boost::filesystem::ifstream infile(filename);
  XMLTag tag=parse_tag(infile,true);
  if (tag.name!="JOB")
    boost::throw_exception(std::runtime_error("missing <JOB> element in jobfile"));
  tag=parse_tag(infile);
  if (tag.name=="OUTPUT") {
    if(tag.attributes["file"]!="")
      outfilepath=boost::filesystem::complete(
               boost::filesystem::path(tag.attributes["file"],
               boost::filesystem::native),filename.branch_path());
    else
      boost::throw_exception(std::runtime_error(
               "missing 'file' attribute in <OUTPUT> element in jobfile"));
    tag=parse_tag(infile);
    if (tag.name=="/OUTPUT")
      tag=parse_tag(infile);
  }
  // make output path absolute
  while (tag.name=="TASK") {
    if (tag.attributes["status"]=="" || tag.attributes["status"]=="new")
      taskstatus.push_back(TaskNotStarted);
    else if (tag.attributes["status"]=="running")
      taskstatus.push_back(TaskHalted);
    else if (tag.attributes["status"]=="finished")
      taskstatus.push_back(TaskFinished);
    else
      boost::throw_exception(std::runtime_error(
               "illegal status attribute in <TASK> element in jobfile"));
    tag=parse_tag(infile);
    CheckpointFiles files;
    if (tag.name=="INPUT") {
      files.in=boost::filesystem::path(tag.attributes["file"],
               boost::filesystem::native);
      if (files.in.empty())
        boost::throw_exception(std::runtime_error(
               "missing 'file' attribute in <INPUT> element in jobfile"));
      tag=parse_tag(infile);
      if (tag.name=="/INPUT")
        tag=parse_tag(infile);
    }
    else
      boost::throw_exception(std::runtime_error(
               "missing <INPUT> element in jobfile"));
    if (tag.name=="OUTPUT") {
      files.out=boost::filesystem::path(tag.attributes["file"],
               boost::filesystem::native);
      if (files.out.empty())
        boost::throw_exception(std::runtime_error(
               "missing 'file' attribute in <OUTPUT> element in jobfile"));
      tag=parse_tag(infile);
      if (tag.name=="/OUTPUT")
        tag=parse_tag(infile);
    }
    if (files.out.empty())
      files.out=files.in;
    files.in=boost::filesystem::complete(files.in,filename.branch_path());
    if (tag.name!="/TASK")
      boost::throw_exception(std::runtime_error(
               "missing </TASK> tag in jobfile"));
    tag = parse_tag(infile);
    taskfiles.push_back(files);
  }
  if (tag.name!="/JOB")
    boost::throw_exception(std::runtime_error("missing </JOB> tag in jobfile"));
}

// reload a simulation, this time to perform actual work
void  MasterScheduler::remake_task(ProcessList& where, const int i)
{
  if(tasks[i]==0)
    boost::throw_exception(std::logic_error(
               "cannot remake a simulation that does not exist"));
  delete tasks[i];
  tasks[i]=make_task(where,taskfiles[i].in);
}

MasterScheduler::~MasterScheduler()
{
  for (int i=0;i<tasks.size();++i)
    if(tasks[i])
      delete tasks[i];
  if(processes.size()>1) {
    OMPDump dump;
    dump.send(processes,MCMP_stop_slave_scheduler);
  }
}

void MasterScheduler::checkpoint()
{
  bool make_backup=boost::filesystem::exists(outfilepath);
  boost::filesystem::path filename=outfilepath;
  boost::filesystem::path dir=outfilepath.branch_path();
  
  if (make_backup)
    filename=dir/(filename.leaf()+".bak");
  { // scope for out
    oxstream out(filename);

    out << header("UTF-8") << stylesheet(xslt_path("job.xsl"));
    out << start_tag("JOB") 
        << xml_namespace("xsi","http://www.w3.org/2001/XMLSchema-instance")
        << attribute("xsi:noNamespaceSchemaLocation",
               "http://xml.comp-phys.org/2003/8/job.xsd");
    int local_sim=-1;
    
    for (int i=0; i<tasks.size();i++) {
      if (taskstatus[i]==TaskFinished) {
        out << start_tag("TASK") << attribute("status","finished")
            << start_tag("INPUT") 
            << attribute("file",taskfiles[i].out.native_file_string())
            << end_tag() << end_tag();
        std::cout  << "Checkpointing task# " << i+1 << "\n";
        if (tasks[i]!=0 && boost::filesystem::complete(taskfiles[i].out,dir).string()!=taskfiles[i].in.string()) {          
          tasks[i]->checkpoint(boost::filesystem::complete(taskfiles[i].out,dir));
          taskfiles[i].in=boost::filesystem::complete(taskfiles[i].out,dir);
        }
        if (tasks[i]!=0) 
          delete tasks[i];
        tasks[i]=0;
      }
      else if(taskstatus[i]==TaskNotExisting) {
        out << start_tag("TASK") << attribute("status","finished")
            << start_tag("INPUT") << attribute("file",taskfiles[i].in.native_file_string())
            << end_tag() << end_tag();
        std::cout  << "Task# " << i+1 << " does not exist\n";
      } 
      else {
        out << start_tag("TASK") 
            << attribute("status",((taskstatus[i]==TaskNotStarted) ? "new" : "running"))
            << start_tag("INPUT") << attribute("file",taskfiles[i].out.native_file_string())
            << end_tag() << end_tag();
        if(theTask != tasks[i]) {
          tasks[i]->checkpoint(boost::filesystem::complete(taskfiles[i].out,dir));
                taskfiles[i].in=boost::filesystem::complete(taskfiles[i].out,dir);
                std::cout  << "Checkpointing task# " << i+1 << "\n";
        }
        else
          local_sim=i;
      }
    }
    if(local_sim>=0) {
      tasks[local_sim]->checkpoint(boost::filesystem::complete(taskfiles[local_sim].out,dir));
      taskfiles[local_sim].in=boost::filesystem::complete(taskfiles[local_sim].out,dir);
      std::cout  << "Checkpointing local task# " << local_sim+1 << "\n";
    }
    out << end_tag("JOB");
  }
  if(make_backup) {
    boost::filesystem::remove(outfilepath);
    boost::filesystem::rename(filename,outfilepath);
  }
}


int MasterScheduler::check_comm_signals(ProcessList& )
{  
  return 0;
}


int MasterScheduler::check_comm_signals()
{
  ProcessList p;
  return check_comm_signals(p);
}


int MasterScheduler::check_signals()
{
  switch(sig())
    {
    case SignalHandler::NOSIGNAL:
        break;
      
    case SignalHandler::USER1:
    case SignalHandler::USER2:
      std::cerr << "Checkpointing...\n";
      checkpoint();
      break;
      
    case SignalHandler::STOP:
      std::cerr  << "Checkpointing and stopping...\n";
      checkpoint();
      sig.stopprocess(); // stop the process
      break;

    case SignalHandler::TERMINATE:
      std::cerr  << "Checkpointing and exiting...\n";
      return SignalHandler::TERMINATE;
      
    default:
      boost::throw_exception ( std::logic_error( "default on switch reached in MasterScheduler::check_signals()"));
    }
  return SignalHandler::NOSIGNAL;
}

// store the results and delete the simulation
void MasterScheduler::finish_task(int i)
{ 
  if (tasks[i] == 0)
    return;
  std::cout  << "Halting Task " << i+1 << ".\n";
  tasks[i]->halt();
  taskstatus[i] = TaskHalted;
  if (make_summary) {
    sim_results[i] = tasks[i]->get_summary();
  }
  std::cerr << "got summary\n"; 
  tasks[i]->checkpoint(boost::filesystem::complete(taskfiles[i].out,outfilepath.branch_path()));
  delete tasks[i];
  tasks[i]=0;
  taskstatus[i] = TaskFinished;      
}

} // namespace scheduler
} // namespace alps
