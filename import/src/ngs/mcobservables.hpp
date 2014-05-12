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

#ifndef ALPS_NGS_MCOBSERVABLES_HPP
#define ALPS_NGS_MCOBSERVABLES_HPP

#include <alps/ngs/config.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/mcobservable.hpp>

#include <alps/alea/observable_fwd.hpp>

#include <map>
#include <string>

namespace alps {
    class ALPS_DECL mcobservables : public std::map<std::string, mcobservable> {

        public: 

            mcobservable & operator[](std::string const & name);

            mcobservable const & operator[](std::string const & name) const;

            bool has(std::string const & name) const;

            void insert(std::string const & name, mcobservable obs);

            void insert(std::string const & name, Observable const * obs);

            void reset(bool equilibrated = false);

            void save(hdf5::archive & ar) const;

            void load(hdf5::archive & ar);

            void merge(mcobservables const &);

            void output(std::ostream & os) const;

            // DO NOT USE! use set << ngs::RealObservable(name);
            // TODO: shold be solved using friends
            void create_RealObservable(std::string const & name, uint32_t binnum = 0);

            // DO NOT USE! use set << ngs::RealVectorObservable(name);
            // TODO: shold be solved using friends
            void create_RealVectorObservable(std::string const & name, uint32_t binnum = 0);

            // DO NOT USE! use set << ngs::SimpleRealObservable(name);
            // TODO: shold be solved using friends
            void create_SimpleRealObservable(std::string const & name);

            // DO NOT USE! use set << ngs::SimpleRealVectorObservable(name);
            // TODO: shold be solved using friends
            void create_SimpleRealVectorObservable(std::string const & name);

            // DO NOT USE! use set << ngs::SignedRealObservable(name, sign);
            // TODO: shold be solved using friends
            void create_SignedRealObservable(std::string const & name, std::string sign = "Sign", uint32_t binnum = 0);

            // DO NOT USE! use set << ngs::SignedRealVectorObservable(name, sign);
            // TODO: shold be solved using friends
            void create_SignedRealVectorObservable(std::string const & name, std::string sign = "Sign", uint32_t binnum = 0);

            // DO NOT USE! use set << ngs::SignedSimpleRealObservable(name, sign);
            // TODO: shold be solved using friends
            void create_SignedSimpleRealObservable(std::string const & name, std::string sign = "Sign");

            // DO NOT USE! use set << ngs::SignedSimpleRealVectorObservable(name, sign);
            // TODO: shold be solved using friends
            void create_SignedSimpleRealVectorObservable(std::string const & name, std::string sign = "Sign");

            // DO NOT USE! use set << ngs::RealTimeSeriesObservable(name);
            // TODO: shold be solved using friends
            void create_RealTimeSeriesObservable(std::string const & name);
            // DO NOT USE! use set << ngs::RealTimeSeriesObservable(name);
            // TODO: shold be solved using friends
            void create_RealVectorTimeSeriesObservable(std::string const & name);
    };

    std::ostream & operator<<(std::ostream & os, mcobservables const & observables);
}

#endif
