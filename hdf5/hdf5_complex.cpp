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

#include <complex>
#include <alps/hdf5.hpp>
#include <vector>
#include <iostream>
#include <algorithm>

template<class T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const & v)
{
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " "));
    return os;
}


struct foo {
  
    std::complex<double> scalar;
    std::vector<std::complex<double> > vec;
    
    void serialize(alps::hdf5::iarchive & ar)
    {
        ar >> alps::make_pvp("scalar", scalar);
        ar >> alps::make_pvp("vector", vec);
    }
    void serialize(alps::hdf5::oarchive & ar) const
    {
        ar << alps::make_pvp("scalar", scalar);
        ar << alps::make_pvp("vector", vec);
    }
    
};
int main () {
    
    foo b;
    b.scalar = std::complex<double>(3,4);
    b.vec = std::vector<std::complex<double> >(5, std::complex<double>(0,7));
    {
        alps::hdf5::oarchive ar("test.h5");
        ar << alps::make_pvp("/test/foo", b);
    }
    
    // check
    {
        foo t_b;
        alps::hdf5::iarchive ar("test.h5");
        ar >> alps::make_pvp("/test/foo", t_b);
        std::cout << "scalar (write): " << b.scalar << std::endl;
        std::cout << "scalar (read): " << t_b.scalar << std::endl;
        std::cout << "vector (write): " << b.vec << std::endl;
        std::cout << "vector (read): " << t_b.vec << std::endl;
    }
    
}
