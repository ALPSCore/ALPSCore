/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_HDF5_PYTHON_CPP
#define ALPS_NGS_HDF5_PYTHON_CPP

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/cast.hpp>
#include <alps/hdf5/pair.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>

#include <alps/ngs/boost_python.hpp>

#include <boost/scoped_ptr.hpp>

#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/object.hpp>
#include <boost/python/numeric.hpp>

#include <string>
#include <iterator>
#include <stdexcept>

namespace alps {
    namespace hdf5 {

        namespace detail {

            template<> struct is_vectorizable<boost::python::object> {
                static bool apply(boost::python::object const & value);
            };

            template<> struct get_extent<boost::python::object> {
                static std::vector<std::size_t> apply(boost::python::object const & value);
            };

            template<> struct set_extent<boost::python::object> {
                static void apply(boost::python::object & value, std::vector<std::size_t> const & extent);
            };
        }

        ALPS_DECL void save(
              archive & ar
            , std::string const & path
            , boost::python::object const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        );

        ALPS_DECL void load(
              archive & ar
            , std::string const & path
            , boost::python::object & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        );
        
        namespace detail {

            template<> struct is_vectorizable<boost::python::list> {
                static bool apply(boost::python::list const & value);
            };

            template<> struct get_extent<boost::python::list> {
                static std::vector<std::size_t> apply(boost::python::list const & value);
            };

            template<> struct set_extent<boost::python::list> {
                static void apply(boost::python::list & value, std::vector<std::size_t> const & extent);
            };
        }

        ALPS_DECL void save(
              archive & ar
            , std::string const & path
            , boost::python::list const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        );

        ALPS_DECL void load(
              archive & ar
            , std::string const & path
            , boost::python::list & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        );

        namespace detail {

            template<> struct is_vectorizable<boost::python::numeric::array> {
                static bool apply(boost::python::numeric::array const & value);
            };

            template<>  struct get_extent<boost::python::numeric::array> {
                static std::vector<std::size_t> apply(boost::python::numeric::array const & value);
            };

            template<>  struct set_extent<boost::python::numeric::array> {
                // To set the extent of a numpy array, we need the type, extent is set in load
                static void apply(boost::python::numeric::array & value, std::vector<std::size_t> const & extent);
            };
        }

        ALPS_DECL void save(
              archive & ar
            , std::string const & path
            , boost::python::numeric::array const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        );

        ALPS_DECL void load(
              archive & ar
            , std::string const & path
            , boost::python::numeric::array & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        );
        
        ALPS_DECL void save(
              archive & ar
            , std::string const & path
            , boost::python::dict const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        );
        
        ALPS_DECL void load(
              archive & ar
            , std::string const & path
            , boost::python::dict & value
            , std::vector<std::size_t> chunk = std::vector<std::size_t>()
            , std::vector<std::size_t> offset = std::vector<std::size_t>()
        );
    }
}

#endif
