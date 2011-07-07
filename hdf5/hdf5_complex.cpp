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