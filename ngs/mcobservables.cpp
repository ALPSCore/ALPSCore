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

#include <alps/ngs/macros.hpp>
#include <alps/ngs/mcobservables.hpp>

#include <alps/hdf5.hpp>
#include <alps/alea/observable.h>
#include <alps/alea/observableset.h>

#include <stdexcept>

namespace alps {

    mcobservable & mcobservables::operator[](std::string const & name) {
        if (!has(name))
            ALPS_NGS_THROW_OUT_OF_RANGE("No observable found with the name: " + name);
        return std::map<std::string, mcobservable>::find(name)->second;
    }

    mcobservable const & mcobservables::operator[](std::string const & name) const {
        if (!has(name))
            ALPS_NGS_THROW_OUT_OF_RANGE("No observable found with the name: " + name);
        return std::map<std::string, mcobservable>::find(name)->second;
    }

    bool mcobservables::has(std::string const & name) const {
        return std::map<std::string, mcobservable>::find(name) != std::map<std::string, mcobservable>::end();
    }

    void mcobservables::insert(std::string const & name, mcobservable obs) {
        if (has(name))
            ALPS_NGS_THROW_OUT_OF_RANGE("There exists alrady a observable with the name: " + name);
        std::map<std::string, mcobservable>::insert(make_pair(name, obs));
    }

    void mcobservables::insert(std::string const & name, Observable const * obs) {
        insert(name, mcobservable(obs));
    }

    void mcobservables::reset(bool equilibrated) {
        for(std::map<std::string, mcobservable>::iterator it = std::map<std::string, mcobservable>::begin(); it != std::map<std::string, mcobservable>::end(); ++it)
            it->second.get_impl()->reset(equilibrated);
    }

    void mcobservables::serialize(hdf5::iarchive & ar)  {
        ObservableSet set;
        ar >> make_pvp("/simulation/realizations/0/clones/0/results", set);
        for(ObservableSet::const_iterator it = set.begin(); it != set.end(); ++it)
            if (has(it->first))
                operator[](it->first) = mcobservable(it->second);
            else
                insert(it->first, it->second);
        for(ObservableSet::const_iterator it = set.begin(); it != set.end(); ++it)
            if (it->second->is_signed())
                operator[](it->first).get_impl()->set_sign(*(operator[](it->second->sign_name()).get_impl()));
    }

    void mcobservables::serialize(hdf5::oarchive & ar) const {
        for(std::map<std::string, mcobservable>::const_iterator it = std::map<std::string, mcobservable>::begin(); it != std::map<std::string, mcobservable>::end(); ++it)
            ar
                << make_pvp(ar.encode_segment(it->first), it->second)
            ;
    }

    void mcobservables::output(std::ostream & os) const {
        for(std::map<std::string, mcobservable>::const_iterator it = std::map<std::string, mcobservable>::begin(); it != std::map<std::string, mcobservable>::end(); ++it)
            std::cout << std::fixed << std::setprecision(5) << it->first << ": " << it->second << std::endl;
    }

    void mcobservables::create_RealObservable(std::string const & name) {
        insert(name, new RealObservable(name));
    }

    void mcobservables::create_RealVectorObservable(std::string const & name) {
        insert(name, new RealVectorObservable(name));
    }

    void mcobservables::create_SimpleRealObservable(std::string const & name) {
        insert(name, new SimpleRealObservable(name));
    }

    void mcobservables::create_SimpleRealVectorObservable(std::string const & name) {
        insert(name, new SimpleRealVectorObservable(name));
    }

    void mcobservables::create_SignedRealObservable(std::string const & name, std::string sign) {
        insert(name, new SignedObservable<RealObservable>(name));
        operator[](name).get_impl()->set_sign(*(operator[](sign).get_impl()));
    }

    void mcobservables::create_SignedRealVectorObservable(std::string const & name, std::string sign) {
        insert(name, new SignedObservable<RealVectorObservable>(name));
        operator[](name).get_impl()->set_sign(*(operator[](sign).get_impl()));
    }

    void mcobservables::create_SignedSimpleRealObservable(std::string const & name, std::string sign) {
        insert(name, new SignedObservable<SimpleRealObservable>(name));
        operator[](name).get_impl()->set_sign(*(operator[](sign).get_impl()));
    }

    void mcobservables::create_SignedSimpleRealVectorObservable(std::string const & name, std::string sign) {
        insert(name, new SignedObservable<SimpleRealVectorObservable>(name));
        operator[](name).get_impl()->set_sign(*(operator[](sign).get_impl()));
    }

    std::ostream & operator<<(std::ostream & os, mcobservables const & results) {
        results.output(os);
        return os;
    }

}
