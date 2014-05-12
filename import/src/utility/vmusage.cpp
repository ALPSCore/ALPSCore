/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010-2012 by Haruhiko Matsuo <halm@looper.t.u-tokyo.ac.jp>,
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

// reference: http://d.hatena.ne.jp/naoya/20080727/1217119867

#include "vmusage.hpp"
#include <alps/config.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <string>
#include <vector>
#ifdef ALPS_HAVE_UNISTD_H
# include <unistd.h> // getpid
#else
  int getpid() { return -1; }
#endif

namespace alps {

vmusage_type vmusage(int pid) {
  vmusage_type usage;
  usage["Pid"] = 0;
  usage["VmPeak"] = 0;
  usage["VmSize"] = 0;
  usage["VmHWM"] = 0;
  usage["VmRSS"] = 0;

  if (pid == -1) pid = getpid();
  if (pid < 0) return usage;
  usage["Pid"] = pid;

  // open /proc/${PID}/status
  std::string file =
    std::string("/proc/") + boost::lexical_cast<std::string>(pid) + std::string("/status");
  std::ifstream fin(file.c_str());
  if (fin.fail()) return usage;

  // read VmPeak, etc (in [kB])
  do {
    std::string line;
    getline(fin, line);
    std::vector<std::string> words;
    boost::split(words, line, boost::is_space());
    if (words.size() >= 2) {
      if (words[0] == "Pid:")
        usage["Pid"] = boost::lexical_cast<unsigned long>(words[words.size()-1]);
    }
    if (words.size() >= 3) {
      if (words[0] == "VmPeak:")
        usage["VmPeak"] = boost::lexical_cast<unsigned long>(words[words.size()-2]);
      else if (words[0] == "VmSize:")
        usage["VmSize"] = boost::lexical_cast<unsigned long>(words[words.size()-2]);
      else if (words[0] == "VmHWM:")
        usage["VmHWM"] = boost::lexical_cast<unsigned long>(words[words.size()-2]);
      else if (words[0] == "VmRSS:")
        usage["VmRSS"] = boost::lexical_cast<unsigned long>(words[words.size()-2]);
    }
  } while (fin.good());
  return usage;
}

} // end namespace alps
