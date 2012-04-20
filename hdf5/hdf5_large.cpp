#include <alps/hdf5.hpp>

#include <boost/filesystem.hpp>

#include <vector>

using namespace std;
using namespace alps;

int main () {

    if (boost::filesystem::exists(boost::filesystem::path("large0.h5")))
        boost::filesystem::remove(boost::filesystem::path("large0.h5"));
    if (boost::filesystem::exists(boost::filesystem::path("large1.h5")))
        boost::filesystem::remove(boost::filesystem::path("large1.h5"));

	hdf5::archive ar("large%d.h5", "al");
	for (unsigned long long s = 1; s < (1ULL << 29); s <<= 1) {
		std::cout << s << std::endl;
		vector<double> vec(s, 10.);
		ar << make_pvp("/" + cast<std::string>(s), vec);
	}

    if (boost::filesystem::exists(boost::filesystem::path("large0.h5")))
        boost::filesystem::remove(boost::filesystem::path("large0.h5"));
    if (boost::filesystem::exists(boost::filesystem::path("large1.h5")))
        boost::filesystem::remove(boost::filesystem::path("large1.h5"));
}
