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

#include <alps/hdf5.hpp>
#include <alps/ngs/mcobservable.hpp>

#include <alps/config.h>
#include <alps/alea/observable_fwd.hpp>

#include <map>
#include <string>

namespace alps {

    class mcobservables : public std::map<std::string, mcobservable> {

        public: 

            mcobservable & operator[](std::string const & name);

            mcobservable const & operator[](std::string const & name) const;

            bool has(std::string const & name) const;

            void insert(std::string const & name, mcobservable obs);

            void insert(std::string const & name, Observable const * obs);

            void reset(bool equilibrated = false);

            void save(hdf5::archive & ar) const;

            void load(hdf5::archive & ar);

            void output(std::ostream & os) const;

            void create_RealObservable(std::string const & name);

            void create_RealVectorObservable(std::string const & name);

            void create_SimpleRealObservable(std::string const & name);

            void create_SimpleRealVectorObservable(std::string const & name);

            void create_SignedRealObservable(std::string const & name, std::string sign = "Sign");

            void create_SignedRealVectorObservable(std::string const & name, std::string sign = "Sign");

            void create_SignedSimpleRealObservable(std::string const & name, std::string sign = "Sign");

            void create_SignedSimpleRealVectorObservable(std::string const & name, std::string sign = "Sign");

    };

    std::ostream & operator<<(std::ostream & os, mcobservables const & observables);
}

#endif