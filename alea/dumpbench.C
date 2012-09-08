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

#include <alps/alea.h>
#include <alps/osiris/xdrdump.h> 
#include <alps/hdf5.hpp>

#include <boost/filesystem.hpp>
#include <boost/random.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <string>
#include <iostream>

#include <hdf5.h>

int main() {
  int count = 100;
  int size = 10000;
  
  std::string const xdr_filename = "dumpbench.dump";
  std::string const hdf5_filename = "dumpbench.h5";
  if (boost::filesystem::exists(boost::filesystem::path(xdr_filename)))
    boost::filesystem::remove(boost::filesystem::path(xdr_filename));
  if (boost::filesystem::exists(boost::filesystem::path(hdf5_filename)))
    boost::filesystem::remove(boost::filesystem::path(hdf5_filename));

  alps::ObservableSet measurement;
  {
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    boost::minstd_rand0 engine;
    boost::uniform_01<boost::minstd_rand0> random(engine);
    measurement //<< alps::make_observable(alps::RealObservable("Test"), true)
                << alps::RealObservable("Sign")
                << alps::RealObservable("No Measurements")
                << alps::IntHistogramObservable("Histogram", 0, 10)
                << alps::RealObservable("Test 2")
                << alps::RealObservable("Test 3")
//                << alps::RealVectorObservable("Test 4")
    ;
//     std::valarray<double> vec;
//     vec.resize(1000);
     for (int i = 0; i < 1000000; ++i) {
//        vec[i % vec.size()] = random();
//      measurement["Test"] << random();
      measurement["Sign"] << 1.0;
      measurement["Histogram"] << static_cast<int>(10*random());
      measurement["Test 2"] << random();
      measurement["Test 3"] << random();
//      measurement["Test 4"] << vec;
    }
    alps::RealObsevaluator e2 = measurement["Test 2"];
    alps::RealObsevaluator e3 = measurement["Test 3"];
    alps::RealObsevaluator ratio("Ratio");
    ratio = e2 / e3;
//    measurement.addObservable(ratio);
    boost::posix_time::ptime stop = boost::posix_time::microsec_clock::local_time();
    std::cerr << "Generating mesurement            : " << 0.001 * (stop - start).total_milliseconds() << " sec\n";
  }
  
  {
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    alps::hdf5::archive oar(hdf5_filename, "a");
    for (int c = 0; c < count; ++c) {
      oar << make_pvp("/test/" + boost::lexical_cast<std::string>(c) + "/result", measurement);
    }
    boost::posix_time::ptime stop = boost::posix_time::microsec_clock::local_time();
    std::cerr << "Writing to HDF5                  : " << 0.001 * (stop - start).total_milliseconds() << " sec\n";
  }

  {
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    alps::OXDRFileDump dump=alps::OXDRFileDump(boost::filesystem::path(xdr_filename));
    for (int c = 0; c < count; ++c) {
      dump << measurement;
    }
    boost::posix_time::ptime stop = boost::posix_time::microsec_clock::local_time();
    std::cerr << "Writing to XDR                   : " << 0.001 * (stop - start).total_milliseconds()<< " sec\n";
  }

  measurement.clear();
  {
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    alps::hdf5::archive iar(hdf5_filename, "r");
    for (int c = 0; c < count; ++c) {
      iar >> make_pvp("/test/" + boost::lexical_cast<std::string>(c) + "/result", measurement);
    }
    boost::posix_time::ptime stop = boost::posix_time::microsec_clock::local_time();
    std::cerr << "Reading from HDF5                : " << 0.001 * (stop - start).total_milliseconds() << " sec\n";
  }

  measurement.clear();
  {
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    alps::IXDRFileDump dump=alps::IXDRFileDump(boost::filesystem::path(xdr_filename));
    for (int c = 0; c < count; ++c) {
      dump >> measurement;
    }
    boost::posix_time::ptime stop = boost::posix_time::microsec_clock::local_time();
    std::cerr << "Reading from XDR                 : " << 0.001 * (stop - start).total_milliseconds() << " sec\n";
  }

  boost::filesystem::remove(boost::filesystem::path(hdf5_filename));
  boost::filesystem::remove(boost::filesystem::path(xdr_filename));

  {
    std::vector<double> data(size);
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    alps::OXDRFileDump dump=alps::OXDRFileDump(boost::filesystem::path(xdr_filename));
    for (int c = 0; c < count; ++c) {
      dump << data;
    }
    boost::posix_time::ptime stop = boost::posix_time::microsec_clock::local_time();
    std::cerr << "Writing Vector to XDR            : " << 0.001 * (stop - start).total_milliseconds() << " sec\n";
  }
  boost::filesystem::remove(boost::filesystem::path(xdr_filename));

  {
	using namespace alps;
	std::vector<double> data(size);
	alps::hdf5::archive oar(hdf5_filename, "a");
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
	{
		for (int c = 0; c < count; ++c) {
		  oar << make_pvp("/vec", data);
		}
	}
    boost::posix_time::ptime stop = boost::posix_time::microsec_clock::local_time();
    std::cerr << "Writing Vector to HDF5 Archive   : " << 0.001 * (stop - start).total_milliseconds() << " sec\n";
    start = boost::posix_time::microsec_clock::local_time();
	{
		for (int c = 0; c < count; ++c)
			oar.write("/vec", &data.front(), std::vector<std::size_t>(1, data.size()), std::vector<std::size_t>(1, data.size()), std::vector<std::size_t>(1, 0));
	}
    stop = boost::posix_time::microsec_clock::local_time();
	std::cerr << "Writing Pointer to HDF5 Archive  : " << 0.001 * (stop - start).total_milliseconds() << " sec\n";
  }

  {
	using namespace alps;
	std::vector<double> data;
	alps::hdf5::archive iar(hdf5_filename, "r");
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
	{
		for (int c = 0; c < count; ++c)
		  iar >> make_pvp("/vec", data);
	}
    boost::posix_time::ptime stop = boost::posix_time::microsec_clock::local_time();
    std::cerr << "Reading Vector from HDF5 Archive : " << 0.001 * (stop - start).total_milliseconds() << " sec\n";
  }

  boost::filesystem::remove(boost::filesystem::path(hdf5_filename));
  {
	std::vector<double> data(size);
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
	hsize_t hsize = size;
	hid_t fileId = H5Fcreate(hdf5_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    for (int c = 0; c < count; ++c) {
		hid_t dataspaceId = H5Screate_simple(1, &hsize, NULL);
		hid_t datatypeId = H5Tcopy(H5T_NATIVE_DOUBLE);
		hid_t datasetId = H5Dcreate2(fileId, "/vec", datatypeId, dataspaceId, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		herr_t status = H5Dwrite(datasetId, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data.front());
		H5Sclose(dataspaceId);
		H5Tclose(datatypeId);
		H5Dclose(datasetId);
	}
	H5Fflush(fileId, H5F_SCOPE_GLOBAL);
	H5Fclose(fileId);
    boost::posix_time::ptime stop = boost::posix_time::microsec_clock::local_time();
    std::cerr << "Writing Vector to HDF5 Native    : " << 0.001 * (stop - start).total_milliseconds() << " sec\n";
  }
  boost::filesystem::remove(boost::filesystem::path(hdf5_filename));
  return 0;
}
