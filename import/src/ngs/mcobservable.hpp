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

#ifndef ALPS_NGS_MCOBSERVABLE_HPP
#define ALPS_NGS_MCOBSERVABLE_HPP

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/config.hpp>

#include <alps/alea/observable_fwd.hpp>

#include <map>
#include <iostream>

namespace alps {

    class ALPS_DECL mcobservable {

        public:

            mcobservable();
            mcobservable(Observable const * obs);
            mcobservable(mcobservable const & rhs);

            virtual ~mcobservable();

            mcobservable & operator=(mcobservable rhs);

            Observable * get_impl();

            Observable const * get_impl() const;

            std::string const & name() const;

            template<typename T> mcobservable & operator<<(T const & value);

            void save(hdf5::archive & ar) const;
            void load(hdf5::archive & ar);

            void merge(mcobservable const &);

            void output(std::ostream & os) const;

        private:

            Observable * impl_;
            static std::map<Observable *, std::size_t> ref_cnt_;

    };

    std::ostream & operator<<(std::ostream & os, mcobservable const & obs);

}

#endif
