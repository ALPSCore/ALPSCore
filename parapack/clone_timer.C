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

#include <alps/parapack/clone_timer.h>
#include <iostream>

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif

  double progress = 0;
  alps::clone_timer timer(boost::posix_time::milliseconds(100), progress);

  alps::clone_timer::loops_t loops = 1024;
  alps::clone_timer::time_t start = timer.current_time();
  alps::clone_timer::time_t end = start + boost::posix_time::seconds(1);
  std::cerr << "start time = " << start << std::endl;

  while (true) {
    std::cerr << "loops = " << loops << std::endl;
    for (alps::clone_timer::loops_t i = 0; i < loops; ++i) {
      timer.current_time();
    }
    alps::clone_timer::time_t current = timer.current_time();
    std::cerr << "current time = " << current << " (" << (current - start) << ")\n";
    if (current > end) break;
    loops = timer.next_loops(loops, current);
  }
  end += boost::posix_time::seconds(1);
  while (true) {
    std::cerr << "loops = " << loops << std::endl;
    for (alps::clone_timer::loops_t i = 0; i < loops; ++i) {
      timer.current_time();
      timer.current_time();
      timer.current_time();
      timer.current_time();
    }
    alps::clone_timer::time_t current = timer.current_time();
    std::cerr << "current time = " << current << " (" << (current - start) << ")\n";
    if (current > end) break;
    loops = timer.next_loops(loops, current);
  }

#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& exp) {
  std::cerr << exp.what() << std::endl;
  std::abort();
}
#endif
  return 0;
}
