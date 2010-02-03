/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010 by Synge Todo <wistaria@comp-phys.org>
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

#include <iostream>
#include <alps/alea.h>
#include <boost/filesystem.hpp>
#include <boost/random.hpp>

int main()
{
  std::string file = "alea2.h5";

  typedef boost::minstd_rand0 random_base_type;
  typedef boost::uniform_01<random_base_type> random_type;
  random_base_type random_int;
  random_type random(random_int);

  alps::ObservableSet measurement;
  measurement << alps::make_observable(alps::SimpleRealObservable("Test"), true)
              << alps::RealObservable("Sign")
              << alps::RealObservable("No Measurements");
  for (int i = 0; i < 10000; ++i) {
    measurement["Test"] << random();
    measurement["Sign"] << 1.0;
  }
  boost::filesystem::remove(boost::filesystem::path(file));
  {
    alps::hdf5::oarchive h5(file);
    h5 << make_pvp("/test/0/result", measurement);
  }
  std::cout << "+ + + + + + + + +" << std::endl;
  alps::ObservableSet measurement2;
  {
    alps::hdf5::iarchive h5(file);
    h5 >> make_pvp("/test/0/result", measurement2);
  }
  std::cout << measurement2;

  boost::filesystem::remove(boost::filesystem::path(file));
}
