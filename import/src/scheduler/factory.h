/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_SCHEDULER_FACTORY_H
#define ALPS_SCHEDULER_FACTORY_H

#include <alps/scheduler/task.h>
#include <alps/config.h>
#include <boost/filesystem/path.hpp>
#include <iostream>

namespace alps {
namespace scheduler {

//=======================================================================
// Factory
//
// a factory for user defined task and subtask objects
//-----------------------------------------------------------------------

class ALPS_DECL Factory
{
public:
  Factory() {}
  virtual ~Factory() {}
  virtual Task* make_task(const ProcessList&,const boost::filesystem::path&) const;
  virtual Task* make_task(const ProcessList&,const boost::filesystem::path&,const Parameters&) const;
  virtual Task* make_task(const ProcessList&,const Parameters&) const;
  virtual Worker* make_worker(const ProcessList&,const Parameters&,int) const;
  virtual void print_copyright(std::ostream&) const=0;
};

template <class TASK>
class SimpleFactory : public Factory
{
public:
  SimpleFactory() {}
  
  Task* make_task(const ProcessList& w,const boost::filesystem::path& fn) const
  {
    return new TASK(w,fn);
  }

  Task* make_task(const ProcessList& w,const Parameters& p) const
  {
    return new TASK(w,p);
  }
  
  void print_copyright(std::ostream& out) const
  {
    TASK::print_copyright(out);
  }
};

template <class TASK, class WORKER>
class BasicFactory : public SimpleFactory<TASK>
{
public:
  BasicFactory() {}
  
  Worker* make_worker(const ProcessList& where ,const Parameters& parms,int node) const
  {
    return new WORKER(where,parms,node);
  }

  void print_copyright(std::ostream& out) const
  {
    SimpleFactory<TASK>::print_copyright(out);
    WORKER::print_copyright(out);
  }
};

} // namespace scheduler
} // namespace alps
 
#endif
