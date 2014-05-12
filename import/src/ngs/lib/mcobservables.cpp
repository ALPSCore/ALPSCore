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
#include <alps/ngs/stacktrace.hpp>

#include <alps/ngs/mcobservables.hpp>

#include <alps/alea/observable.h>
#include <alps/alea/observableset.h>

#include <boost/make_shared.hpp>

#include <stdexcept>

namespace alps {
    
    mcobservable & mcobservables::operator[](std::string const & name) {
        iterator it = find(name);
        if (it == end())
            throw std::out_of_range("No observable found with the name: " + name + ALPS_STACKTRACE);
        return it->second;
    }

    mcobservable const & mcobservables::operator[](std::string const & name) const {
        const_iterator it = find(name);
        if (it == end())
            throw std::out_of_range("No observable found with the name: " + name + ALPS_STACKTRACE);
        return it->second;
    }

    bool mcobservables::has(std::string const & name) const {
        return std::map<std::string, mcobservable>::find(name) != std::map<std::string, mcobservable>::end();
    }

    void mcobservables::insert(std::string const & name, mcobservable obs) {
        if (has(name))
            throw std::out_of_range("There exists alrady a observable with the name: " + name + ALPS_STACKTRACE);
        std::map<std::string, mcobservable>::insert(make_pair(name, obs));
    }

    void mcobservables::insert(std::string const & name, Observable const * obs) {
        insert(name, mcobservable(obs));
    }

    void mcobservables::reset(bool equilibrated) {
        for(std::map<std::string, mcobservable>::iterator it = std::map<std::string, mcobservable>::begin(); it != std::map<std::string, mcobservable>::end(); ++it)
            it->second.get_impl()->reset(equilibrated);
    }

    void mcobservables::save(hdf5::archive & ar) const {
        for(std::map<std::string, mcobservable>::const_iterator it = std::map<std::string, mcobservable>::begin(); it != std::map<std::string, mcobservable>::end(); ++it)
            ar
                << make_pvp(ar.encode_segment(it->first), it->second)
            ;
    }

    void mcobservables::load(hdf5::archive & ar)  {
        ObservableSet set;
        ar >> make_pvp(ar.get_context(), set);
        for(ObservableSet::const_iterator it = set.begin(); it != set.end(); ++it)
            if (has(it->first))
                operator[](it->first) = mcobservable(it->second);
            else
                insert(it->first, it->second);
        for(ObservableSet::const_iterator it = set.begin(); it != set.end(); ++it)
            if (it->second->is_signed())
                operator[](it->first).get_impl()->set_sign(*(operator[](it->second->sign_name()).get_impl()));
    }

    void mcobservables::merge(mcobservables const & arg) {
        for (std::map<std::string, mcobservable>::const_iterator it = arg.begin(); it != arg.end(); ++it)
            if (has(it->first))
                std::map<std::string, mcobservable>::find(it->first)->second.merge(it->second);
            else
                insert(it->first, it->second);
    }

    void mcobservables::output(std::ostream & os) const {
        for(std::map<std::string, mcobservable>::const_iterator it = std::map<std::string, mcobservable>::begin(); it != std::map<std::string, mcobservable>::end(); ++it)
            os << std::fixed << std::setprecision(5) << it->first << ": " << it->second << std::endl;
    }

    void mcobservables::create_RealObservable(std::string const & name, uint32_t binnum) {
        insert(name, boost::make_shared<RealObservable>(name,binnum).get());
    }

    void mcobservables::create_RealVectorObservable(std::string const & name, uint32_t binnum) {
        insert(name, boost::make_shared<RealVectorObservable>(name,binnum).get());
    }

    void mcobservables::create_SimpleRealObservable(std::string const & name) {
        insert(name, boost::make_shared<SimpleRealObservable>(name).get());
    }

    void mcobservables::create_SimpleRealVectorObservable(std::string const & name) {
        insert(name, boost::make_shared<SimpleRealVectorObservable>(name).get());
    }

    void mcobservables::create_SignedRealObservable(std::string const & name, std::string sign, uint32_t binnum) {
        insert(name, boost::make_shared<SignedObservable<RealObservable> >(name, binnum).get());
        if (find(sign) == end())
            throw std::runtime_error("the sign " +  sign + " does not exist" + ALPS_STACKTRACE);
        operator[](name).get_impl()->set_sign(*(operator[](sign).get_impl()));
    }

    void mcobservables::create_SignedRealVectorObservable(std::string const & name, std::string sign, uint32_t binnum) {
        insert(name, boost::make_shared<SignedObservable<RealVectorObservable> >(name, binnum).get());
        if (find(sign) == end())
            throw std::runtime_error("the sign " +  sign + " does not exist" + ALPS_STACKTRACE);
        operator[](name).get_impl()->set_sign(*(operator[](sign).get_impl()));
    }

    void mcobservables::create_SignedSimpleRealObservable(std::string const & name, std::string sign) {
        insert(name, boost::make_shared<SignedObservable<SimpleRealObservable> >(name).get());
        if (find(sign) == end())
            throw std::runtime_error("the sign " +  sign + " does not exist" + ALPS_STACKTRACE);
        operator[](name).get_impl()->set_sign(*(operator[](sign).get_impl()));
    }

    void mcobservables::create_SignedSimpleRealVectorObservable(std::string const & name, std::string sign) {
        insert(name, boost::make_shared<SignedObservable<SimpleRealVectorObservable> >(name).get());
        if (find(sign) == end())
            throw std::runtime_error("the sign " +  sign + " does not exist" + ALPS_STACKTRACE);
        operator[](name).get_impl()->set_sign(*(operator[](sign).get_impl()));
    }

    void mcobservables::create_RealTimeSeriesObservable(std::string const & name) {
        insert(name, boost::make_shared<RealTimeSeriesObservable>(name).get());
    }

    std::ostream & operator<<(std::ostream & os, mcobservables const & results) {
        results.output(os);
        return os;
    }

}
