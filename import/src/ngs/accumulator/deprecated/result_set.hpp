/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_ALEA_RESULT_SET_HPP
#define ALPS_NGS_ALEA_RESULT_SET_HPP

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/alea/wrapper/result_wrapper.hpp>

#include <map>
#include <string>

namespace alps {
    namespace accumulator {

        class ALPS_DECL result_set {

            public: 
                typedef std::map<std::string, boost::shared_ptr<detail::result_wrapper> > map_type;

                typedef map_type::iterator iterator;
                typedef map_type::const_iterator const_iterator;

                detail::result_wrapper & operator[](std::string const & name);

                detail::result_wrapper const & operator[](std::string const & name) const;

                bool has(std::string const & name) const;

                void insert(std::string const & name, boost::shared_ptr<detail::result_wrapper> ptr);

                void save(hdf5::archive & ar) const;

                void load(hdf5::archive & ar);

                void merge(result_set const &);

                void print(std::ostream & os) const;

                iterator begin();
                iterator end();
                const_iterator begin() const;
                const_iterator end() const;
                void clear();

            private:
                map_type storage;
        };

        inline std::ostream & operator<<(std::ostream & out, result_set const & arg) {
            arg.print(out);
            return out;
        }

    }
}

#endif
