
#include <alps/alea.h>

int main(int argc, char** argv)
{
    try {
        
        alps::ObservableSet measurements_;
        measurements_ << alps::RealObservable("E");
        
        alps::hdf5::archive ar("test_observableset.h5", alps::hdf5::archive::WRITE);
        measurements_.get<alps::RealObservable>("E") << 1;
        ar << alps::make_pvp("/simulation/results/", measurements_);
        
    } catch (std::exception & e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    
    return 0;
}
