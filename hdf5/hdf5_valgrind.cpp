#include <iostream>
#include <sstream>
#include <vector>
#include <alps/hdf5.hpp>

int main() {

    for (int i=0; i<100; ++i) {
        std::vector<double> vec(10,2.);
        alps::hdf5::archive ar("test.h5", "w");
        std::ostringstream ss;
        ss << "/vec" << i;
        ar << alps::make_pvp(ss.str(), vec);
    }
}
