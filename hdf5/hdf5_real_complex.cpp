#include <iostream>
#include <complex>

#include <vector>
#include <alps/numeric/matrix.hpp>
#include <alps/utility/vectorio.hpp>
#include <alps/hdf5.hpp>

template <class T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& v)
{
	os << "[" <<alps::write_vector(v, " ", 6) << "]";
	return os;
}

int main() {
	
	const int vsize = 6, msize=4;
	
	
	std::vector<double> v(vsize, 3.2);
	alps::numeric::matrix<double> A(msize,msize, 1.5);
	
	std::cout << "v: " << v << std::endl;

	{
		alps::hdf5::archive ar("real_complex.h5", "w");
		ar["/matrix"] << A;
		ar["/vec"] << v;
	}
	
	std::vector<std::complex<double> > w;
	alps::numeric::matrix<std::complex<double> > B;
	{
		alps::hdf5::archive ar("real_complex.h5", "r");
		ar["/matrix"] >> B;
		ar["/vec"] >> w;
	}
	
	std::cout << "w: " << w << std::endl;

	
	bool passed = true;
	
	{
		bool vpassed = true;
		for (int i=0; vpassed && i<vsize; ++i)
			vpassed = (v[i] == w[i]);
		passed = passed && vpassed;
 	}

	{
		bool vpassed = true;
		for (int i=0; vpassed && i<msize; ++i)
			for (int j=0; vpassed && j<msize; ++j)
				vpassed = (A(i,j) == B(i,j));
		passed = passed && vpassed;
 	}
	
	
	return (passed) ? 0 : 1;
}