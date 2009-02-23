#include "mocasito/io/container.hpp"
#include "mocasito/io/hdf5.hpp"
#include <complex>
#include <iostream>
#include <cassert>

const std::size_t len = 20;

template<typename Engine, typename T> void test_engine_fixed(std::string const & ifile, std::string const & ofile, std::string const & name) {
	{
		mocasito::io::container<Engine> container(ifile, ofile);
		T value = 20;
		container["/scalar/" + name] = value;
		std::vector<T> v;
		for (std::size_t i = 0; i < len; ++i)
			v.push_back(static_cast<T>(i));
		container["/vectors/" + name] = v;
		container.flush();
	}
	{
		mocasito::io::container<Engine> container(ifile, ofile);
		std::vector<T> w;
		for (std::size_t i = 0; i < len; ++i)
			w.push_back(static_cast<T>(len + i));
		container["/vectors/" + name] << w;
		container.flush();
	}
}
template<typename Engine, typename T> void test_engine_number(std::string const & ifile, std::string const & ofile, std::string const & name) {
	test_engine_fixed<Engine, T>(ifile, ofile, name);
	{
		T value = 20;
		mocasito::io::container<Engine> container(ifile, ofile);
		container["/scalar/@" + name] = value;
		container.flush();
	}
	{
		mocasito::io::container<Engine> container(ifile, ofile);
		std::vector<std::size_t> v;
		assign(v, container["/vectors/" + name]);
		assert(v.size() == 2 * len);
		for (std::size_t i = 0; i < v.size(); ++i)
			assert(v[i] == i);
	}
}
template<typename Engine> void test_engine(std::string const & ifile, std::string const & ofile = "") {
	test_engine_number<Engine, char>(ifile, ofile, "char");
	test_engine_number<Engine, signed char>(ifile, ofile, "signedchar");
	test_engine_number<Engine, unsigned char>(ifile, ofile, "unsignedchar");
	test_engine_number<Engine, short>(ifile, ofile, "short");
	test_engine_number<Engine, signed short>(ifile, ofile, "char");
	test_engine_number<Engine, int>(ifile, ofile, "int");
	test_engine_number<Engine, unsigned>(ifile, ofile, "unsigned");
	test_engine_number<Engine, long>(ifile, ofile, "long");
	test_engine_number<Engine, unsigned long>(ifile, ofile, "unsignedlong");
	test_engine_number<Engine, long long>(ifile, ofile, "longlong");
	test_engine_number<Engine, unsigned long long>(ifile, ofile, "unsignedlonglong");
	test_engine_number<Engine, float>(ifile, ofile, "float");
	test_engine_number<Engine, double>(ifile, ofile, "double");
	test_engine_number<Engine, long double>(ifile, ofile, "longdouble");
	{
		mocasito::io::container<Engine> container(ifile, ofile);
		bool value = true;
		container["/scalar/bool"] = value;
		container["/scalar/@bool"] = value;
		container.flush();
	}
	test_engine_fixed<Engine, std::complex<int> >(ifile, ofile, "complexint");
	test_engine_fixed<Engine, std::complex<double> >(ifile, ofile, "complexdouble");
	{
		int v[] = {1, 4, 2, 5, 8, 5};
		{
			mocasito::io::container<Engine> container(ifile, ofile);
			container["/vectors/fix"] = v;
			container.flush();
		}
		{
			mocasito::io::container<Engine> container(ifile, ofile);
			std::vector<int> u;
			int w[6];
			assign(u, container["/vectors/fix"]);
			assert(u.size() == 6);
			for (std::size_t i = 0; i < 6; ++i)
				assert(v[i] == u[i]);
			assign(w, container["/vectors/fix"]);
			for (std::size_t i = 0; i < 6; ++i)
				assert(v[i] == w[i]);
		}
	}
	/*{
		typedef std::complex<double> nested_t[5];
		{
			mocasito::io::container<Engine> container(ifile, ofile);
			nested_t v, w[23];
			container["/scalar/nested"] = v;
			container["/vectors/nested"] = w;
			container.flush();
		}
	}*/
	{
		mocasito::io::container<Engine> container(ifile, ofile);
		std::vector<int> v;
		container["/vectors/empty"] = v;
		container.flush();
	}
	{
		mocasito::io::container<Engine> container(ifile, ofile);
		std::vector<int> v;
		assign(v, container["/vectors/empty"]);
		assert(v.size() == 0);
	}
	{
		mocasito::io::container<Engine> container(ifile, ofile);
		std::vector<int> v;
		container["/string/std"] = std::string("std");
		container["/string/c"] = "cstr";
		container["/string/empty"] = "";
//		container["/string/@attr"] = std::string("value");
		container.flush();
	}
	{
		mocasito::io::container<Engine> container(ifile, ofile);
		assert(container["/string/std"] == std::string("std"));
		assert(container["/string/std"] == "std");
		assert(container["/string/c"] == std::string("cstr"));
		assert(container["/string/c"] == "cstr");
		assert(container["/string/empty"] == std::string(""));
		assert(container["/string/empty"] == "");
		std::string s = container["/string/std"];
		assert(s == "std");
	}
}
int main(int argc, char ** argv){
	test_engine<mocasito::io::hdf5>(std::string("io.h5"));
}
