#include "staging.h"

#include <string>
#include <fstream>
#include <iostream>

namespace alps {
namespace parapack {

void load_checkpoints(boost::filesystem::path const& file_chp,
		      boost::filesystem::path const& basedir,
		      std::queue<suspended_queue_t>& suspended_queue) {

  std::cout << "  tasks ordered by " << file_chp << " = ";
  FILE* fp;
#ifdef _WIN32
  if ((fp = _wfopen(file_chp.c_str(), L"r")) == NULL) {
#else
  if ((fp = fopen(file_chp.c_str(), "r")) == NULL) {
#endif
      std::cout << "no" << std::endl;
    
  } else {
    std::cout << "yes" << std::endl;
    int task, clone, group;
    while(fscanf(fp, "%d %d %d", &task, &clone, &group) != EOF) {
      suspended_queue.push(boost::tuple<int,int,int>(task, clone, group));
    }
  }
}

}
}

