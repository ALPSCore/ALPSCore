/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2002-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

#ifndef ALPS_SCHEDULER_TASK_H
#define ALPS_SCHEDULER_TASK_H

#include <alps/scheduler/worker.h>
#include <alps/parameters.h>
#include <boost/smart_ptr.hpp>

namespace alps {

namespace scheduler {

struct CheckpointFiles
{
  boost::filesystem::path in;
  boost::filesystem::path out;
};

//=======================================================================
// TaskStatus
//
// a class describing the status of a task, how much work is
// still needed, and where it is running. for use by the scheduler
//-----------------------------------------------------------------------

struct TaskStatus 
{
  int32_t number; // index in the simulation list of the scheduler
  uint32_t cpus; // number of cpus per subtask
  boost::posix_time::ptime next_check; // time of next checking if finished
  double work; // estimate of work still needed
  ProcessList where; // processors on which it is running
  
  TaskStatus()
    : number(-1),
      cpus(1),
      next_check(boost::posix_time::second_clock::local_time()),
      work(-1)
  {
  }
};

//=======================================================================
// AbstractTask
//
// the abstract base class for all classes describing a task
//-----------------------------------------------------------------------

class AbstractTask
{
public:
  AbstractTask();
  AbstractTask(const ProcessList&);
  virtual ~AbstractTask() {}

  virtual void checkpoint(const boost::filesystem::path&) const = 0; 

  virtual uint32_t cpus() const=0; // cpus per run
  virtual bool local() {return false;}; // is it running on the local process?
  
  virtual void add_processes(const ProcessList&);
  virtual void add_process(const Process&) = 0;
  virtual void delete_processes(const ProcessList&);
  virtual void delete_process(const Process&) = 0;
  
  virtual void start() = 0; // start all runs
  virtual void run() = 0; // run for some time (in seconds)
  virtual void halt() = 0; // halt all runs, simulation is finished        

  virtual double work() const {return 1.;}; // return amount of work needed
  virtual bool finished(double&) const = 0; // check if task is finished
  virtual bool handle_message(const Process& master,int tag); // deal with messages

  int finished_notime() const // no time estimate needed
  { 
    double dummy; 
    return finished(dummy);
  }

  // how much work is left?

protected:
  AbstractWorker* theWorker; // the run running on this CPU
  ProcessList where; // the list of work processes for this simulation
};


class Task : public AbstractTask
{
public:
  static void print_copyright(std::ostream&);
  
  Task(const ProcessList&, const boost::filesystem::path&);        
  ~Task();
  
  virtual void construct(); // needs to be called to finish construction

  void checkpoint(const boost::filesystem::path&) const; // write into a file

  void add_process(const Process&);
  void delete_process(const Process&);

  uint32_t cpus() const {return 1;}
  bool local() {return (where.size() ? 1 : 0);} 
  const alps::Parameters& get_parameters() const { return parms;}

  void start(); // start simulation
  void run(); // run a few steps and return control
  virtual void dostep()=0; // do a step
  void finish(); // mark as finished
  bool finished() const { double dummy ; return finished(dummy);}
  bool finished(double&) const; // check if simulation is finished
  bool started() const { return started_;}
  void halt();
  double work() const; // return amount of work needed

protected:
  virtual void write_xml_header(alps::oxstream&) const;
  virtual void write_xml_trailer(alps::oxstream&) const;
  virtual void write_xml_body(alps::oxstream&, const boost::filesystem::path&) const=0;
  virtual void handle_tag(std::istream&, const XMLTag&);

  alps::Parameters parms;
  bool finished_;
  boost::filesystem::path infilename;


private:
  void parse_task_file(bool=false);
  bool started_; // is the task running?
};


class WorkerTask : public Task
{
protected:        
  enum RunStatus {
    RunNotExisting = 0,
    LocalRun = 1,
    RemoteRun = 2,
    RunOnDump = 3
  };

public:
  WorkerTask(const ProcessList&, const boost::filesystem::path&);        
  ~WorkerTask();
  
  void construct(); // needs to be called to finish construction

  void add_process(const Process&);
  void delete_process(const Process&);

  void start(); // start simulation
  void dostep(); // run a few steps and return control
  bool finished(double&) const; // check if simulation is finished
  void halt();
  double work() const; // return amount of work needed
  double work_done() const; // return amount of work done
  std::vector<AbstractWorker*> runs; // the list of all runs

protected:
  virtual std::string worker_tag() const=0;
  void write_xml_body(alps::oxstream&, const boost::filesystem::path&) const;
  void handle_tag(std::istream&, const XMLTag&);
  std::vector<RunStatus> workerstatus;

private:
  mutable time_t start_time; // when as the simulation started?
  mutable double start_work; // how much work was to be done?
  mutable double old_work;
  mutable std::vector<CheckpointFiles> runfiles; 
};


class RemoteTask : public AbstractTask
{
public:
  RemoteTask(const ProcessList&,const boost::filesystem::path&);
  ~RemoteTask();
  void checkpoint(const boost::filesystem::path&) const; // write into a file

  void add_processes(const ProcessList&);
  void add_process(const Process&);
  void delete_processes(const ProcessList&);
  void delete_process(const Process&);

  uint32_t cpus() const;
  bool local() {return false;} // no, remote

  void start(); // run a few steps and return control
  bool finished(double&) const; // check if simulation is finished
  double work() const; // ask for work needed
  void run(); // should not be called
  void halt(); // halt remote simulation
  bool handle_message(const Process& master,int tag);
};


class SlaveTask : public AbstractTask
{
public:
  SlaveTask(const Process&);
  
  virtual void run(); // run a few steps and return control
  virtual void checkpoint(const boost::filesystem::path& fn) const;
  virtual void add_process(const Process& p);
  virtual void delete_process(const Process& p);
  virtual void start();
  virtual double work() const;
  virtual bool finished(double& x) const;
  virtual void halt();
  virtual uint32_t cpus() const;
private:
  bool started;
  Process runmaster;
}; 

} // namespace scheduler

} // namespace alps

#endif
