/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/ngs/alea/result_set.hpp>

namespace alps {
    namespace accumulator {

        detail::result_wrapper & result_set::operator[](std::string const & name) {
            if (!has(name))
                throw std::out_of_range("No observable found with the name: " + name + ALPS_STACKTRACE);
            return *(storage.find(name)->second);
        }

        detail::result_wrapper const & result_set::operator[](std::string const & name) const {
            if (!has(name))
                throw std::out_of_range("No observable found with the name: " + name + ALPS_STACKTRACE);
            return *(storage.find(name)->second);
        }

        bool result_set::has(std::string const & name) const {
            return storage.find(name) != storage.end();
        }

        void result_set::insert(std::string const & name, boost::shared_ptr<detail::result_wrapper> ptr) {
            if (has(name))
                throw std::out_of_range("There exists alrady a result with the name: " + name + ALPS_STACKTRACE);
            storage.insert(make_pair(name, ptr));
        }

        void result_set::save(hdf5::archive & ar) const {}

        void result_set::load(hdf5::archive & ar) {}

        void result_set::merge(result_set const &) {}

        void result_set::print(std::ostream & os) const {
            for (const_iterator it = begin(); it != end(); ++it)
                os << it->first << ": " << *(it->second);
        }

        // map operations
        result_set::iterator result_set::begin() {
            return storage.begin();
        }

        result_set::iterator result_set::end() {
            return storage.end();
        }

        result_set::const_iterator result_set::begin() const {
            return storage.begin();
        }

        result_set::const_iterator result_set::end() const {
            return storage.end();
        }

        void result_set::clear() {
            storage.clear(); //should be ok b/c shared_ptr
        }
    }
}
