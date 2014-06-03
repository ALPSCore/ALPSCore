/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_ALEA_ACCUMULATOR_SET_HEADER
#define ALPS_NGS_ALEA_ACCUMULATOR_SET_HEADER

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/alea/accumulator.hpp>
#include <alps/ngs/alea/wrapper/accumulator_wrapper.hpp>

#include <map>
#include <string>

namespace alps {
    namespace accumulator {

        struct make_accumulator {
            template<typename T>
            make_accumulator(std::string const name, T const & accum): acc_wrapper(accum), name(name) {}

            detail::accumulator_wrapper acc_wrapper;
            std::string name;
        };

        class ALPS_DECL accumulator_set {

            public: 
                typedef std::map<std::string, boost::shared_ptr<detail::accumulator_wrapper> > map_type;

                typedef map_type::iterator iterator;
                typedef map_type::const_iterator const_iterator;

                detail::accumulator_wrapper & operator[](std::string const & name);

                detail::accumulator_wrapper const & operator[](std::string const & name) const;

                bool has(std::string const & name) const;

                void insert(std::string const & name, boost::shared_ptr<detail::accumulator_wrapper> ptr);

                void save(hdf5::archive & ar) const;

                void load(hdf5::archive & ar);

                void reset(bool equilibrated = false);

                //~ template<typename T, typename Features>
                accumulator_set & operator<< (make_accumulator const & make_acc) {
                    insert(make_acc.name, boost::shared_ptr<detail::accumulator_wrapper>(new detail::accumulator_wrapper(make_acc.acc_wrapper)));
                    return *this;
                }

                void merge(accumulator_set const &);

                void print(std::ostream & os) const;

                iterator begin();
                iterator end();
                const_iterator begin() const;
                const_iterator end() const;
                void clear();

            private:
                map_type storage;
        };
    } 
}

#endif
