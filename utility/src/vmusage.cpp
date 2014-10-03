/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

// reference: http://d.hatena.ne.jp/naoya/20080727/1217119867

#include <alps/utility/vmusage.hpp>
#include <alps/config.hpp>
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
