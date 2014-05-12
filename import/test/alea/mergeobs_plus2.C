#include <iostream>
#include <string>
#include <vector>
#include <alps/alea.h>
#include <boost/random.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

#define make_obs(form) \
  do{ \
    alps::RealObsevaluator o( #form ); \
    o = form; \
    obs.addObservable(o); \
  } while (false)

int main(int argc, char **argv) {
  typedef boost::minstd_rand0 random_base_type;
  typedef boost::uniform_01<random_base_type> random_type;
  random_base_type random_int;
  random_type random(random_int);

  const int MCS = 128;
  const int nsets = (argc > 1) ? boost::lexical_cast<int>(argv[1]) : 2;

  std::vector<alps::ObservableSet> obssets(nsets);
  BOOST_FOREACH(alps::ObservableSet &obs, obssets){
    obs << alps::RealObservable("one");
    obs << alps::RealObservable("two");
    obs.reset(true);
    for(int i = 0; i < MCS; ++i) {
      obs["one"] << 1.0 + (2 * random() - 1) * 0.3;
      obs["two"] << 2.0 + (2 * random() - 1) * 0.5;
    }
  }

  // merge
  for(int i = 1; i < nsets; ++i) obssets[0] << obssets[i];
  alps::ObservableSet& obs = obssets[0];

  alps::RealObsevaluator one = obs["one"];
  alps::RealObsevaluator two = obs["two"];
  make_obs(one + two);
  std::cout << obs;

  return 0;
}
