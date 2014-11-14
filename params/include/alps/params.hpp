/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/params/paramvalue.hpp>
#include <alps/params/paramproxy.hpp>
#include <alps/params/paramiterator.hpp>

#ifdef ALPS_HAVE_PYTHON_DEPRECATED
    #include <alps/ngs/boost_python.hpp>
    #include <boost/python/dict.hpp>
#endif

#include <boost/filesystem.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp> 
#include <boost/program_options.hpp>

#ifdef ALPS_HAVE_MPI
    namespace boost{ namespace mpi{ class communicator; } }
#endif

#include <map>
#include <vector>
#include <string>

namespace po=boost::program_options;

namespace alps {

    class ALPS_DECL params : public po::variables_map {

        // typedef std::map<std::string, detail::paramvalue>::value_type iterator_value_type;

        // friend class detail::paramiterator<params, iterator_value_type>;
        // friend class detail::paramiterator<params const, iterator_value_type const>;

        public:
        
            // typedef detail::paramiterator<params, iterator_value_type> iterator;
            // typedef detail::paramiterator<params const, iterator_value_type const> const_iterator;
            // typedef detail::paramproxy value_type;

            /** Default constructor */
            params() {}

            /** Copy constructor */
            params(params const & arg)
                : keys(arg.keys)
                , values(arg.values)
            {}

            /** Constructor from HDF5 archive. (FIXME: Is it now possible?) */
            params(hdf5::archive ar, std::string const & path = "/parameters");

            /** Constructor from parameter file (FIXME: not possible) */
            params(boost::filesystem::path const &);

            #ifdef ALPS_HAVE_PYTHON_DEPRECATED
                params(boost::python::dict const & arg);
                params(boost::python::str const & arg);
            #endif

            /** FIXME: Not needed? Or trivial. */
            std::size_t size() const;

            /** Erase a parameter  */
            void erase(std::string const &);

            /** Access a parameter as boost::any<T> */
            value_type operator[](std::string const &);

            /** Access a parameter as boost::any<T> */
            value_type const operator[](std::string const &) const;

            /** Check if the paramter is defined */
            bool defined(std::string const &) const;

            /// FIXME: not needed
            iterator begin();
            /// FIXME: not needed
            const_iterator begin() const;

            /// FIXME: not needed
            iterator end();
            /// FIXME: not needed
            const_iterator end() const;

            /// Save parameters to HDF5 archive
            void save(hdf5::archive &) const;

            /// Load parameters from HDF5 archive (clearing the object first)
            void load(hdf5::archive &);

            #ifdef ALPS_HAVE_MPI
            /// Broadcast the parameters to all processes
                void broadcast(boost::mpi::communicator const &, int = 0);
            #endif

        private:

            friend class boost::serialization::access;
            
            template<class Archive> void serialize(Archive & ar, const unsigned int) {
                ar & keys
                   & values
                ;
            }

            void setter(std::string const &, detail::paramvalue const &);
      void parse_text_parameters(boost::filesystem::path const & path);

            detail::paramvalue getter(std::string const &);

            std::vector<std::string> keys;
            std::map<std::string, detail::paramvalue> values;
    };

    ALPS_DECL std::ostream & operator<<(std::ostream & os, params const & arg);
}

