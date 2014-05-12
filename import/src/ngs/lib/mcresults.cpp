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

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/mcresults.hpp>
#include <alps/ngs/stacktrace.hpp>

#include <alps/alea/observableset.h>

#include <stdexcept>

namespace alps {

    mcresult & mcresults::operator[](std::string const & name) {
        if (!has(name))
            throw std::out_of_range("No result found with the name: " + name + "\n" + ALPS_STACKTRACE);
        return std::map<std::string, mcresult>::find(name)->second;
    }

    mcresult const & mcresults::operator[](std::string const & name) const {
        if (!has(name))
            throw std::out_of_range("No result found with the name: " + name + "\n" + ALPS_STACKTRACE);
        return std::map<std::string, mcresult>::find(name)->second;
    }

    bool mcresults::has(std::string const & name) const {
        return std::map<std::string, mcresult>::find(name) != std::map<std::string, mcresult>::end();
    }

    void mcresults::insert(std::string const & name, mcresult res) {
        if (has(name))
            throw std::out_of_range("There exists alrady a result with the name: " + name + "\n" + ALPS_STACKTRACE);
        std::map<std::string, mcresult>::insert(make_pair(name, res));
    }

    void mcresults::erase(std::string const & name) {
        if (!has(name))
            throw std::out_of_range("There is no result with the name: " + name + "\n" + ALPS_STACKTRACE);
        std::map<std::string, mcresult>::erase(name);
    }

    void mcresults::save(hdf5::archive & ar) const {
        for(std::map<std::string, mcresult>::const_iterator it = std::map<std::string, mcresult>::begin(); it != std::map<std::string, mcresult>::end(); ++it)
            if (it->second.count())
                ar
                    << make_pvp(ar.encode_segment(it->first), it->second)
                ;
    }

    void mcresults::load(hdf5::archive & ar)  {
        ObservableSet set;
        // TODO: do not use hard coded path!
        ar >> make_pvp("/simulation/realizations/0/clones/0/results", set);
        for(ObservableSet::const_iterator it = set.begin(); it != set.end(); ++it)
            insert(it->first, mcresult(it->second));
    }

    void mcresults::output(std::ostream & os) const {
        for(std::map<std::string, mcresult>::const_iterator it = std::map<std::string, mcresult>::begin(); it != std::map<std::string, mcresult>::end(); ++it)
            os << std::fixed << std::setprecision(5) << it->first << ": " << it->second << std::endl;
    }

    std::ostream & operator<<(std::ostream & os, mcresults const & results) {
        results.output(os);
        return os;
    }

}
