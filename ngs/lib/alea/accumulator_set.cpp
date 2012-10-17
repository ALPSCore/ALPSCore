/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2012 by Mario Koenz <mkoenz@ethz.ch>                       *
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
#include <alps/ngs/accumulator_set.hpp>

namespace alps {
    namespace alea
    {
        detail::accumulator_wrapper & accumulator_set::operator[](std::string const & name)
        {
            if (!has(name))
                throw std::out_of_range("No observable found with the name: " + name + ALPS_STACKTRACE);
            return *(storage.find(name)->second);
        }

        detail::accumulator_wrapper const & accumulator_set::operator[](std::string const & name) const
        {
            if (!has(name))
                throw std::out_of_range("No observable found with the name: " + name + ALPS_STACKTRACE);
            return *(storage.find(name)->second);
        }

        bool accumulator_set::has(std::string const & name) const
        {
            return storage.find(name) != storage.end();
        }
        
        void accumulator_set::insert(std::string const & name, boost::shared_ptr<detail::accumulator_wrapper> ptr)
        {
            if (has(name))
                throw std::out_of_range("There exists alrady an accumulator with the name: " + name + ALPS_STACKTRACE);
            storage.insert(make_pair(name, ptr));
        }

        void accumulator_set::save(hdf5::archive & ar) const
        {
        }

        void accumulator_set::load(hdf5::archive & ar)
        {
        }

        void accumulator_set::merge(accumulator_set const &)
        {
        }

        void accumulator_set::output(std::ostream & os) const
        {
        }
        
        //~ map operations
        accumulator_set::iterator accumulator_set::begin()
        {
            return storage.begin();
        }
        
        accumulator_set::iterator accumulator_set::end()
        {
            return storage.end();
        }
        accumulator_set::const_iterator accumulator_set::begin() const
        {
            return storage.begin();
        }
        
        accumulator_set::const_iterator accumulator_set::end() const
        {
            return storage.end();
        }
        
    } //end namespace alea
} //end namespace alps
