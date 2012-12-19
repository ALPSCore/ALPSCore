/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2009 by Synge Todo <wistaria@comp-phys.org>
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

#include "queue.h"

namespace alps {

bool operator<(task_queue_element_t const& lhs, task_queue_element_t const& rhs) {
  if (lhs.weight < rhs.weight)
    return true;
  else if (lhs.weight > rhs.weight)
    return false;
  else
    return (lhs.task_id > rhs.task_id);
}

bool check_queue_element_t::due() const {
  return boost::posix_time::second_clock::local_time() >= time;
}

bool operator<(check_queue_element_t const& lhs, check_queue_element_t const& rhs) {
  return lhs.time > rhs.time;
}

check_queue_element_t next_taskinfo(boost::posix_time::time_duration const& interval) {
  boost::posix_time::ptime check =  boost::posix_time::second_clock::local_time() + interval;
  // std::cerr << "debug: next taskinfo will be at " << to_simple_string(check) << std::endl;
  return check_queue_element_t(check_type::taskinfo, check, 0, 0, 0);
}

check_queue_element_t next_checkpoint(uint32_t tid, uint32_t cid, uint32_t gid,
  boost::posix_time::time_duration const& interval) {
  boost::posix_time::ptime check =  boost::posix_time::second_clock::local_time() + interval;
  // std::cerr << "debug: next checkpointing of clone[" << tid << "," << cid << "] will be at "
  //           << to_simple_string(check) << std::endl;
  return check_queue_element_t(check_type::checkpoint, check, tid, cid, gid);
}

check_queue_element_t next_report(tid_t tid, cid_t cid, gid_t gid,
  boost::posix_time::time_duration const& interval) {
  boost::posix_time::ptime check =  boost::posix_time::second_clock::local_time() + interval;
  // std::cerr << "debug: next report of clone[" << tid << "," << cid << "] will be at "
  //           << to_simple_string(check) << std::endl;
  return check_queue_element_t(check_type::report, check, tid, cid, gid);
}

check_queue_element_t next_vmusage(boost::posix_time::time_duration const& interval) {
  boost::posix_time::ptime check =  boost::posix_time::second_clock::local_time() + interval;
  return check_queue_element_t(check_type::vmusage, check, 0, 0, 0);
}

} // end namespace alps
