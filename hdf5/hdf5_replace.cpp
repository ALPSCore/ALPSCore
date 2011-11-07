#include <alps/hdf5.hpp>

#include <vector>

using namespace std;

int main () {

   vector<double> vec(100, 10.);

   {
       alps::hdf5::archive h5ar("res.h5", alps::hdf5::archive::WRITE);
       h5ar << alps::make_pvp("/vec2", vec);
   }
   {
       alps::hdf5::archive h5ar("res.h5", alps::hdf5::archive::WRITE | alps::hdf5::archive::REPLACE);
       h5ar << alps::make_pvp("/vec", vec);
   }
   {
       vector<double> tmp;
       alps::hdf5::archive h5ar("res.h5");
       h5ar >> alps::make_pvp("/vec2", tmp);
   }

}
