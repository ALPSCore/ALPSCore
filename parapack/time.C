/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2008 by Synge Todo <wistaria@comp-phys.org>
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

#include <alps/parapack/queue.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/foreach.hpp>
#include <iostream>
#include <queue>

int main()
{
  // check for boost::posix_time

  boost::posix_time::ptime t0 = boost::posix_time::second_clock::local_time();
  std::cerr << "Current time: " << to_simple_string(t0) << std::endl;

  boost::posix_time::ptime t1 = t0 + boost::posix_time::seconds(10);
  std::cerr << "10 seconds after: " << to_simple_string(t1) << std::endl;

  boost::posix_time::ptime t2 = t0 + boost::posix_time::minutes(5);
  std::cerr << "5 minutes after: " << to_simple_string(t2) << std::endl;

  boost::posix_time::ptime t3 = t0 + boost::posix_time::hours(1);
  std::cerr << "One hour after: " << to_simple_string(t3) << std::endl;

  std::cout << "Order comparison: ";
  if (t0 < t1 && t1 < t2 && t2 < t3)
    std::cout << "OK\n";
  else
    std::cout << "error\n";

  std::priority_queue<alps::check_queue_element_t> queue;
  queue.push(alps::next_checkpoint(0, 0, 0, boost::posix_time::seconds(10)));
  queue.push(alps::next_checkpoint(0, 0, 0, boost::posix_time::seconds(2)));
  queue.push(alps::next_checkpoint(0, 0, 0, boost::posix_time::seconds(5)));
  queue.push(alps::next_checkpoint(0, 0, 0, boost::posix_time::seconds(1)));
  std::cerr << to_simple_string(queue.top().time) << std::endl;

  return 0;
}

