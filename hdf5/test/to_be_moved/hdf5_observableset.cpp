/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


#include <alps/alea.h>

int main(int argc, char** argv)
{
    try {
        
        alps::ObservableSet measurements_;
        measurements_ << alps::RealObservable("E");
        
        alps::hdf5::archive ar("test_observableset.h5", "a");
        measurements_.get<alps::RealObservable>("E") << 1;
        ar << alps::make_pvp("/simulation/results/", measurements_);
        
    } catch (std::exception & e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    
    return 0;
}
