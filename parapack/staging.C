#include "staging.h"

#include <string>
#include <fstream>
#include <iostream>

namespace alps {
namespace parapack {

void load_checkpoints(boost::filesystem::path const& file_chp,
		      boost::filesystem::path const& basedir,
		      std::queue<suspended_queue_t>& suspended_queue) {
  FILE* fp;
  if ((fp = fopen(file_chp.c_str(), "r")) == NULL) {
    std::cerr << " can't open " << file_chp << "!" << std::endl;
    
  } else {
    std::cerr << " open " << file_chp << std::endl;
    int task, clone, group;
    while(fscanf(fp, "%d %d %d", &task, &clone, &group) != EOF) {
      suspended_queue.push(boost::tuple<int,int,int>(task, clone, group));
    }
  }
}

}
}
