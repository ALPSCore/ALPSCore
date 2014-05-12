/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                              Matthias Troyer <troyer@comp-phys.org>             *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define PY_ARRAY_UNIQUE_SYMBOL pyngsbase_PyArrayHandle

#include <alps/ngs/boost_python.hpp>

#include <alps/mcbase.hpp>
#include <alps/hdf5/archive.hpp>

#include <alps/python/make_copy.hpp>

#ifdef ALPS_HAVE_MPI
    #include <boost/mpi.hpp>
#endif

#include <boost/bind.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/wrapper.hpp>
#include <boost/python/return_internal_reference.hpp>

namespace alps {

    class pymcbase : public mcbase, public boost::python::wrapper<mcbase> {

        public:

             #ifdef ALPS_HAVE_MPI
                pymcbase(boost::python::dict arg, std::size_t seed_offset = 42, boost::mpi::communicator = boost::mpi::communicator())
                    : mcbase(mcbase::parameters_type(arg), seed_offset)
                {}
            #else
                pymcbase(boost::python::dict arg, std::size_t seed_offset = 42)
                    : mcbase(mcbase::parameters_type(arg), seed_offset)
                {}
            #endif

            void update() {
                this->get_override("update")();
            }
            double fraction_completed() const {
                return this->get_override("fraction_completed")();
            }
            void measure() {
                this->get_override("measure")();
            }

            bool run(boost::python::object stop_callback) {
                return mcbase::run(boost::bind(&pymcbase::run_helper, this, stop_callback));
            }

            results_type collect_results(result_names_type const & names = result_names_type()) {
                return names.size() ? mcbase::collect_results(names) : mcbase::collect_results();
            }

            alps::random01 & get_random() {
                return mcbase::random;
            }

            parameters_type & get_parameters() {
                return mcbase::parameters;
            }

            observable_collection_type & get_measurements() {
                return alps::mcbase::measurements;
            }

        private:

            bool run_helper(boost::python::object stop_callback) {
                return boost::python::call<bool>(stop_callback.ptr());
            }

    };
}

BOOST_PYTHON_MODULE(pyngsbase_c) {

    boost::python::class_<alps::pymcbase, boost::noncopyable>(
          "mcbase",
          #ifdef ALPS_HAVE_MPI
              boost::python::init<boost::python::dict, boost::python::optional<std::size_t, boost::mpi::communicator> >()
          #else
              boost::python::init<boost::python::dict, boost::python::optional<std::size_t> >()
          #endif
    )
        .add_property("random", boost::python::make_function(&alps::pymcbase::get_random, boost::python::return_internal_reference<>()))
        .add_property("parameters", boost::python::make_function(&alps::pymcbase::get_parameters, boost::python::return_internal_reference<>()))
        .add_property("measurements", boost::python::make_function(&alps::pymcbase::get_measurements, boost::python::return_internal_reference<>()))
        .def("run", &alps::pymcbase::run)
        .def("update", boost::python::pure_virtual(&alps::pymcbase::update))
        .def("measure", boost::python::pure_virtual(&alps::pymcbase::measure))
        .def("fraction_completed", boost::python::pure_virtual(&alps::pymcbase::fraction_completed))
        .def("save", static_cast<void(alps::pymcbase::*)(alps::hdf5::archive &) const>(&alps::pymcbase::save))
        .def("load", static_cast<void(alps::pymcbase::*)(alps::hdf5::archive &)>(&alps::pymcbase::load))
    ;

}
