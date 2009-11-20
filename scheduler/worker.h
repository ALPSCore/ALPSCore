/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_SCHEDULER_WORKER_H
#define ALPS_SCHEDULER_WORKER_H

#include <alps/config.h>
#include <alps/scheduler/info.h>
#include <alps/parameter.h>
#include <alps/random/rngfactory.h>
#include <alps/h5archive.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/random.hpp>
#include <boost/filesystem/path.hpp>
#include <alps/osiris/process.h>
#include <alps/osiris/dump.h>
#include <cmath>
#include <iostream>

namespace alps {

namespace scheduler {

namespace {
  const double MCTimeFactor = 0.05;
}

typedef struct rt {
  double T;
  double mean;
  double error;
  double count;

  rt operator+=(const rt c) {
    using std::sqrt;
    if (T != c.T) 
      std::cerr << "\nname or temperature of summaries to add don't match!!\n";
    if (count == 0)
      return c;
    if (c.count == 0)
      return (*this);
    double newCount = count+c.count;
    mean = (mean*count+c.mean*c.count)/newCount;
    double tmp1 = error*count;
    double tmp2 = c.error*c.count;
    error = sqrt(tmp1*tmp1 + tmp2*tmp2)/newCount;
    count = newCount;
    return (*this);
  }
} ResultType;

typedef std::vector<ResultType> ResultsType;

//=======================================================================
// AbstractWorker
//
// the abstract base class for all classes describing a Monte Carlo run
// a run is an actual simulation, running on one CPU.
// the collection of all runs with the same parameters is called
// simulation.
//-----------------------------------------------------------------------

class AbstractWorker
{
public:                
  AbstractWorker() {};
  virtual ~AbstractWorker() {};
  virtual void save_to_file(const boost::filesystem::path&) const=0;
  virtual void load_from_file(const boost::filesystem::path&)=0;
  virtual void set_parameters(const Parameters& parms)=0;
  virtual TaskInfo get_info() const = 0;
  virtual double work_done() const =0;
  virtual void start_worker() = 0;
  virtual void halt_worker() = 0;
  virtual bool handle_message(const Process& runmaster,int32_t tag) =0;
  virtual ResultType get_summary() const = 0;
};

//=======================================================================
// Worker
//
// base class for the actual run class which will be implemented by the
// user.
//-----------------------------------------------------------------------


class ALPS_DECL Worker : public AbstractWorker
{
public:
  Worker(const ProcessList&,const Parameters&, int32_t=0);
  Worker(const Parameters&, int32_t=0);
  virtual ~Worker();
  void set_parameters(const Parameters& parms);
  virtual bool change_parameter(const std::string& name, const StringValue& value);
  virtual void save_worker(ODump&) const;
  virtual void load_worker(IDump&);
	#ifdef ALPS_HAVE_HDF5
		virtual void save_worker(h5archive &) const;
//		virtual void load_worker(h5archive &);
	#endif
  virtual void write_xml(const boost::filesystem::path& name, const boost::filesystem::path& ckpt_name="") const;
  void save_to_file(const boost::filesystem::path&) const;
  void load_from_file(const boost::filesystem::path&);
  // creates a new information object containing information about this run
  TaskInfo get_info() const;
  void start_worker();
  virtual void halt_worker();
  virtual void start() {}
  virtual void halt() {}
  virtual std::string work_phase();
  void change_phase(const std::string&);
  virtual void run();
  bool handle_message(const Process& runmaster,int32_t tag);
  virtual void dostep();
  double work_done() const;
  virtual ResultType get_summary() const;  

protected:
  int32_t version;
  int32_t user_version;

  typedef buffered_rng_base engine_type;
  mutable boost::shared_ptr<engine_type> engine_ptr;
  mutable boost::variate_generator<engine_type&, boost::uniform_real<> > random;
  mutable boost::variate_generator<engine_type&, boost::uniform_real<> > random_01;

  double random_real(double a=0., double b=1.) { return a+b*random();}
  //return boost::variate_generator<random_type&,boost::uniform_real<> >(random,boost::uniform_real<>(a,b))();
  int random_int(int a, int b) 
  { return a+int((b-a+1)*random());}
  //{ return boost::variate_generator<random_type&,boost::uniform_int<> >(random,boost::uniform_int<>(a,b))();}
  int random_int(int n) 
  { return int(n*random());}
  //{ return random_int(0,n-1);}

  int node;
  Parameters parms;
  ProcessList where;
private:
  TaskInfo info;
  int halted,started;
};


//=======================================================================
// RemoteWorker
//
// this class is just a class for proxy objects. The actual run is on
// another node/computer, messages are sent to the actual object
// and the return message relayed to the caller.
// Allows for transparent access to remote objects
//-----------------------------------------------------------------------

class RemoteWorker : public AbstractWorker
{
public:
  // constructors/destructor also constructs/destroys actual object
  RemoteWorker(const ProcessList&,const Parameters&, int32_t=0);
  virtual ~RemoteWorker();

  void set_parameters(const Parameters& parms);

  // save function also saves actual object
  void save_to_file(const boost::filesystem::path&) const;
  void load_from_file(const boost::filesystem::path&);
  void start_worker();
  virtual void halt_worker();
  
  virtual TaskInfo get_info() const;
  double work_done() const;
  
  virtual ResultType get_summary() const;

  const Process& process() const {return where;}
  bool handle_message(const Process& runmaster,int32_t tag);
private:
  Process where;
};

} // namespace scheduler

} // namespace alps

#endif 
