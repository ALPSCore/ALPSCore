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

#include <alps/scheduler/factory.h>

namespace alps {
namespace scheduler {

Worker* Factory::make_worker(const ProcessList&,const Parameters&,int) const
{
  boost::throw_exception(std::logic_error("Factory::make_worker() needs to be implemented"));
  return 0;
}

Task* Factory::make_task(const ProcessList& w,const boost::filesystem::path& fn) const
{
  alps::Parameters parms;
  { // scope to close file
    boost::filesystem::ifstream infile(fn);
    parms.extract_from_xml(infile);
  }
  return make_task(w,fn,parms);
}

Task* Factory::make_task(const ProcessList&,const boost::filesystem::path&,const Parameters&) const
{
  boost::throw_exception(std::logic_error("Factory::make_task(const ProcessList&,const boost::filesystem::path&,const Parameters&) needs to be implemented"));
  return 0;
}

Task* Factory::make_task(const ProcessList&,const Parameters&) const
{
  boost::throw_exception(std::logic_error("Factory::make_task(const ProcessList&,const Parameters&) needs to be implemented"));
  return 0;
}

} // namespace scheduler
} // namespace alps
