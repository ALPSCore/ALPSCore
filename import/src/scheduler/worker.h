/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_SCHEDULER_WORKER_H
#define ALPS_SCHEDULER_WORKER_H

#include <alps/config.h>
#include <alps/scheduler/info.h>
#include <alps/parameter.h>
#include <alps/random.h>
#include <boost/smart_ptr.hpp>
#include <boost/filesystem/path.hpp>
#include <alps/osiris/process.h>
#include <alps/osiris/dump.h>
#include <cmath>
#include <iostream>

#include <alps/hdf5.hpp>

namespace alps {

namespace scheduler {

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

class AbstractWorker {
public:                
  AbstractWorker() {};
  virtual ~AbstractWorker() {};
  
  virtual void save(hdf5::archive &) const {};
  virtual void load(hdf5::archive &) {};

  virtual void save_to_file(const boost::filesystem::path&,const boost::filesystem::path&) const=0;
  virtual void load_from_file(const boost::filesystem::path&,const boost::filesystem::path&)=0;
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
  
  virtual void save(hdf5::archive &) const;
  virtual void load(hdf5::archive &);

  virtual void write_xml(const boost::filesystem::path& name) const;
  void save_to_file(const boost::filesystem::path&,const boost::filesystem::path&) const;
  void load_from_file(const boost::filesystem::path&,const boost::filesystem::path&);
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

  typedef buffered_rng_base engine_type;
  mutable boost::shared_ptr<engine_type> engine_ptr;
  mutable boost::variate_generator<engine_type&, boost::uniform_real<> > random;
  mutable boost::variate_generator<engine_type&, boost::uniform_real<> > random_01;

private:
  std::string rng_name() const { return parms.value_or_default("RNG","mt19937");}
  TaskInfo info;
  int halted,started;
  std::string file_name1, file_name2;
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
  void save_to_file(const boost::filesystem::path&,const boost::filesystem::path&) const;
  void load_from_file(const boost::filesystem::path&,const boost::filesystem::path&);
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
