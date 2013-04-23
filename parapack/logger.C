/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2013 by Synge Todo <wistaria@comp-phys.org>
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

#include "logger.h"
#include <boost/lexical_cast.hpp>
#include <boost/date_time.hpp>

namespace alps {

std::string logger::header() {
  return std::string("[") + to_simple_string(boost::posix_time::second_clock::local_time()) + "]: ";
}
std::string logger::task(alps::tid_t tid) {
  return std::string("task[") + boost::lexical_cast<std::string>(tid+1) + ']';
}
std::string logger::clone(alps::tid_t tid, alps::cid_t cid) {
  return std::string("clone[") + boost::lexical_cast<std::string>(tid+1) + ',' +
    boost::lexical_cast<std::string>(cid+1) + ']';
}
std::string logger::group(alps::process_group g) {
  return std::string("processgroup[") + boost::lexical_cast<std::string>(g.group_id+1) + ']';
}
std::string logger::group(alps::thread_group g) {
  return std::string("threadgroup[") + boost::lexical_cast<std::string>(g.group_id+1) + ']';
}

std::string logger::usage(alps::vmusage_type const& u) {
  return std::string("Process ID = ") +
    boost::lexical_cast<std::string>(u.find("Pid")->second) + ", " +
    "VmPeak = " + boost::lexical_cast<std::string>(u.find("VmPeak")->second) + " [kB], " +
    "VmSize = " + boost::lexical_cast<std::string>(u.find("VmSize")->second) + " [kB], " +
    "VmHWM = " + boost::lexical_cast<std::string>(u.find("VmHWM")->second) + " [kB], " +
    "VmRSS = " + boost::lexical_cast<std::string>(u.find("VmRSS")->second) + " [kB]";
}

} // namespace alps
