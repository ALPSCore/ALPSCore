/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

// this must be first
#include <alps/utilities/boost_python.hpp>


#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/pair.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>

#include <alps/utilities/cast.hpp>
#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/numpy_import.hpp>

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

            template<> struct is_vectorizable<boost::python::tuple> {
                static bool apply(boost::python::tuple const & value);
            };

            template<> struct get_extent<boost::python::tuple> {
                static std::vector<std::size_t> apply(boost::python::tuple const & value);
            };
        }

        ALPS_DECL void save(
              archive & ar
            , std::string const & path
            , boost::python::tuple const & value
            , std::vector<std::size_t> size = std::vector<std::size_t>()
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
