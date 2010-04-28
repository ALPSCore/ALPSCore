#include <string>
#include <iostream>
#include <algorithm>
#include <utility>
#include <alps/hdf5.hpp>
#include <alps/utility/encode.hpp>
#include <boost/filesystem.hpp>

const int length = 15;
typedef enum { A, B } enum_type;

namespace alps {
	namespace hdf5 {
		template <> oarchive & serialize(oarchive & ar, std::string const & p, enum_type const & v) {
			ar << alps::make_pvp(p, v == A ? 0 : 1);
			return ar;
		}
		template <> iarchive & serialize(iarchive & ar, std::string const & p, enum_type & v) {
			int t;
			ar >> alps::make_pvp(p, t);
			v = t ? A : B;
			return ar;
		}
	}
}

int main() {
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::cout << alps::hdf5_name_encode("a/b/c&c/d/e") << " " 
				  << alps::hdf5_name_decode(alps::hdf5_name_encode("a/b/c&c/d/e")) << std::endl;
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		int d = 42;
		h5ar << alps::make_pvp("/int", d);
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		int d = 0;
		h5ar >> alps::make_pvp("int", d);
		std::cout << d << std::endl;
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		std::string s;
		h5ar >> alps::make_pvp("int", s);
		std::cout << s << std::endl;
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		int *d = new int[length];
		std::fill_n(d, length, 42);
		h5ar << alps::make_pvp("/foo/bar", d, length);
		delete[] d;
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		int *d = new int[length];
		std::fill_n(d, length, 0);
		h5ar >> alps::make_pvp("/foo/bar", d, length);
		std::copy (d, d + length, std::ostream_iterator<int, char, std::char_traits<char> >(std::cout, " "));
		std::cout << std::endl;
		delete[] d;
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::vector<int> d(length, 42);
		h5ar << alps::make_pvp("/foo/bar2", d);
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		std::vector<int> d;
		h5ar >> alps::make_pvp("/foo/bar2", d);
		std::copy (d.begin(), d.end(), std::ostream_iterator<int, char, std::char_traits<char> >(std::cout, " "));
		std::cout << std::endl;
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		h5ar.delete_data("/foo/bar2");
		std::cout << h5ar.is_data("/foo/bar2") << std::endl;
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::string s("blub");
		h5ar << alps::make_pvp("/foo/bar3", s);
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		std::string s;
		h5ar >> alps::make_pvp("/foo/bar3", s);
		std::cout << s << std::endl;
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		int a = 42;
		h5ar << alps::make_pvp("/foo/@bar4", a);
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		int a = 0;
		h5ar >> alps::make_pvp("/foo/@bar4", a);
		std::cout << a << std::endl;
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::string s("blub");
		h5ar << alps::make_pvp("/foo/@bar5", s);
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		std::string s;
		h5ar >> alps::make_pvp("/foo/@bar5", s);
		std::cout << s << std::endl;
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::complex<double> *d = new std::complex<double>[length];
		h5ar << alps::make_pvp("test/data", d, length);
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		std::complex<double> *d = new std::complex<double>[length];
		h5ar >> alps::make_pvp("test/data", d, length);
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		enum_type d = A;
		h5ar << alps::make_pvp("test/enum", d);
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		enum_type d;
		h5ar >> alps::make_pvp("test/enum", d);
		std::cout << (d == A ? "A" : "B") << std::endl;
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::vector<std::string> d;
		d.push_back("value1");
		d.push_back("val2");
		d.push_back("v3");
		d.push_back("value4");
		h5ar << alps::make_pvp("test/vectorstring", d);
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		std::vector<std::string> d;
		h5ar >> alps::make_pvp("test/vectorstring", d);
		for (std::vector<std::string>::const_iterator it = d.begin(); it != d.end(); ++it)
			std::cout << *it << std::endl;
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::vector<int> d;
		h5ar << alps::make_pvp("/test/growing", d);
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::vector<int> d(length, 42);
		h5ar << alps::make_pvp("/test/growing", d);
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::vector<std::vector<int> > d;
		d.push_back(std::vector<int>(3, 2));
		d.push_back(std::vector<int>(3, 4));
		d.push_back(std::vector<int>(3, 0));
		d.push_back(std::vector<int>(3, 3));
		h5ar << alps::make_pvp("/test/vector", d);
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		std::vector<std::vector<int> > d;
		h5ar >> alps::make_pvp("/test/vector", d);
		for (std::vector<std::vector<int> >::const_iterator it = d.begin(); it != d.end(); ++it)
			std::cout << it->size() << " " << (it->size() ? it->front() : 0) << std::endl;
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::vector<std::valarray<int> > d;
		d.push_back(std::valarray<int>(2, 2));
		d.push_back(std::valarray<int>(4, 2));
		d.push_back(std::valarray<int>(0, 2));
		h5ar << alps::make_pvp("/test/vecval", d);
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		std::vector<std::valarray<int> > d;
		h5ar >> alps::make_pvp("/test/vecval", d);
		for (std::vector<std::valarray<int> >::const_iterator it = d.begin(); it != d.end(); ++it)
			std::cout << it->size() << " " << (it->size() ? (*it)[0] : 0) << std::endl;
	}
/*	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::map<std::string, std::vector<int> > d;
		d["foo"] = std::vector<int>(1, 2);
		d["bar"] = std::vector<int>();
		h5ar << alps::make_pvp("/test/map", d);
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		std::map<std::string, std::vector<int> > d;
		h5ar >> alps::make_pvp("/test/map", d);
		for (std::map<std::string, std::vector<int> >::const_iterator it = d.begin(); it != d.end(); ++it)
			std::cout << it->first << " " << it->second.size() << std::endl;
	}
*/	{
		alps::hdf5::oarchive h5ar("test.h5");
		h5ar.serialize("/path/to/group/to/set/attr");
		h5ar << alps::make_pvp("/path/to/group/to/set/attr/@version", 1);
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::pair<double *, std::vector<std::size_t> > data(std::make_pair(new double[4], std::vector<std::size_t>(2)));
		data.second[0] = data.second[1] = 2;
		h5ar << alps::make_pvp("/test/scalarpair", data);
		delete[] data.first;
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		std::pair<double *, std::vector<std::size_t> > data(NULL, h5ar.extent("/test/scalarpair"));
		data.first = new double[std::accumulate(data.second.begin(), data.second.end(), 1, std::multiplies<std::size_t>())];
		h5ar >> alps::make_pvp("/test/scalarpair", data);
		std::cout << data.second.size() << std::endl;
		delete[] data.first;
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::pair<std::complex<double> *, std::vector<std::size_t> > data(std::make_pair(new std::complex<double>[4], std::vector<std::size_t>(2)));
		data.second[0] = data.second[1] = 2;
		h5ar << alps::make_pvp("/test/complexpair", data);
		delete[] data.first;
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		std::pair<std::complex<double> *, std::vector<std::size_t> > data(NULL, h5ar.extent("/test/complexpair"));
		data.first = new std::complex<double>[std::accumulate(data.second.begin(), data.second.end(), 1, std::multiplies<std::size_t>())];
		h5ar >> alps::make_pvp("/test/complexpair", data);
		std::cout << data.second.size() << std::endl;
		delete[] data.first;
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::complex<double> data[2][2][500];
		h5ar << alps::make_pvp("/test/complexvector", &data[0][0][0], 2 * 2 * 500);
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		std::complex<double> data[2][2][500];
		h5ar >> alps::make_pvp("/test/complexvector", &data[0][0][0], 2 * 2 * 500);
	}
	{
		alps::hdf5::oarchive h5ar("test.h5");
		std::complex<double> data;
		h5ar << alps::make_pvp("/test/complexvalue", data);
	}
	{
		alps::hdf5::iarchive h5ar("test.h5");
		std::complex<double> data;
		h5ar >> alps::make_pvp("/test/complexvalue", data);
	}
	boost::filesystem::remove(boost::filesystem::path("test.h5"));
}
