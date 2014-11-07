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

#ifdef ALPS_HAVE_MPI
    namespace boost{ namespace mpi{ class communicator; } }
#endif

#include <map>
#include <vector>
#include <string>

namespace alps {

    class ALPS_DECL params {

        typedef std::map<std::string, detail::paramvalue>::value_type iterator_value_type;

        friend class detail::paramiterator<params, iterator_value_type>;
        friend class detail::paramiterator<params const, iterator_value_type const>;

        public:

            typedef detail::paramiterator<params, iterator_value_type> iterator;
            typedef detail::paramiterator<params const, iterator_value_type const> const_iterator;
            typedef detail::paramproxy value_type;

            params() {}

            params(params const & arg)
                : keys(arg.keys)
                , values(arg.values)
            {}

            params(hdf5::archive ar, std::string const & path = "/parameters");

            params(boost::filesystem::path const &);

            #ifdef ALPS_HAVE_PYTHON_DEPRECATED
                params(boost::python::dict const & arg);
                params(boost::python::str const & arg);
            #endif

            std::size_t size() const;

            void erase(std::string const &);

            value_type operator[](std::string const &);

            value_type const operator[](std::string const &) const;

            bool defined(std::string const &) const;

            iterator begin();
            const_iterator begin() const;

            iterator end();
            const_iterator end() const;

            void save(hdf5::archive &) const;

            void load(hdf5::archive &);

            #ifdef ALPS_HAVE_MPI
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

