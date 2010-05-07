#include <alps/hdf5.hpp>
#include <alps/utility/encode.hpp>
#include <alps/alea.h>

#include <boost/filesystem.hpp>
#include <boost/random.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

int main() {
    {
        boost::minstd_rand0 engine;
        boost::uniform_01<boost::minstd_rand0> random(engine);
        alps::ObservableSet measurement;
        measurement << alps::make_observable(alps::SimpleRealObservable("Test"), true)
                    << alps::RealObservable("Sign")
                    << alps::RealObservable("No Measurements");
        for (int i = 0; i < 10000; ++i) {
            measurement["Test"] << random();
            measurement["Sign"] << 1.0;
        }
        alps::hdf5::oarchive oar("alea.h5");
        oar << make_pvp("/test/0/result", measurement);
    }
    {
        alps::ObservableSet measurement;
        measurement << alps::make_observable(alps::SimpleRealObservable("Test"), true)
                    << alps::RealObservable("Sign")
                    << alps::RealObservable("No Measurements");
        alps::hdf5::iarchive iar("alea.h5");
        iar >> make_pvp("/test/0/result", measurement);
        std::cout << measurement;
    }
    {
        alps::ObservableSet measurement;
        alps::hdf5::iarchive iar("alea.h5");
        iar >> make_pvp("/test/0/result", measurement);
        std::cout << measurement;
    }
    boost::filesystem::remove(boost::filesystem::path("alea.h5"));
}