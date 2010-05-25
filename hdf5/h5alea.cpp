/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010 by Lukas Gamper <gamperl -at- gmail.com>
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
