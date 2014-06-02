/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/ngs/alea/accumulator_set.hpp>

namespace alps {
    namespace accumulator {

        detail::accumulator_wrapper & accumulator_set::operator[](std::string const & name) {
            if (!has(name))
                throw std::out_of_range("No observable found with the name: " + name + ALPS_STACKTRACE);
            return *(storage.find(name)->second);
        }

        detail::accumulator_wrapper const & accumulator_set::operator[](std::string const & name) const {
            if (!has(name))
                throw std::out_of_range("No observable found with the name: " + name + ALPS_STACKTRACE);
            return *(storage.find(name)->second);
        }

        bool accumulator_set::has(std::string const & name) const{
            return storage.find(name) != storage.end();
        }
        
        void accumulator_set::insert(std::string const & name, boost::shared_ptr<alps::accumulator::detail::accumulator_wrapper> ptr){
            if (has(name))
                throw std::out_of_range("There exists alrady an accumulator with the name: " + name + ALPS_STACKTRACE);
            storage.insert(make_pair(name, ptr));
        }

        void accumulator_set::save(hdf5::archive & ar) const {
            for(const_iterator it = begin(); it != end(); ++it)
                ar[it->first] = *(it->second);
        }

        void accumulator_set::load(hdf5::archive & ar) {}

        void accumulator_set::merge(accumulator_set const &) {}

        void accumulator_set::print(std::ostream & os) const {}

        void accumulator_set::reset(bool equilibrated) {
            for(iterator it = begin(); it != end(); ++it)
                it->second->reset();
        }
        
        //~ map operations
        accumulator_set::iterator accumulator_set::begin() {
            return storage.begin();
        }

        accumulator_set::iterator accumulator_set::end() {
            return storage.end();
        }

        accumulator_set::const_iterator accumulator_set::begin() const {
            return storage.begin();
        }
        
        accumulator_set::const_iterator accumulator_set::end() const {
            return storage.end();
        }
        
        void accumulator_set::clear() {
            storage.clear(); //should be ok b/c shared_ptr
        }
    }
}
