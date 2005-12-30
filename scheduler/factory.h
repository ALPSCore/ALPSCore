/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

#ifndef ALPS_SCHEDULER_FACTORY_H
#define ALPS_SCHEDULER_FACTORY_H

#include <alps/scheduler/task.h>
#include <boost/filesystem/path.hpp>
#include <iostream>

namespace alps {
namespace scheduler {

//=======================================================================
// Factory
//
// a factory for user defined task and subtask objects
//-----------------------------------------------------------------------

class Factory
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
  
  void print_copyright(std::ostream& out) const
  {
    TASK::print_copyright(out);
  }
};

} // namespace scheduler
} // namespace alps
 
#endif
