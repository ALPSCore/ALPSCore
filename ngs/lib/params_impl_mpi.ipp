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

#ifndef ALPS_NGS_PARAMS_IMPL_MPI_IPP
#define ALPS_NGS_PARAMS_IMPL_MPI_IPP

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/param.hpp>
#include <alps/ngs/hdf5/map.hpp>
#include <alps/ngs/detail/params_impl_base.hpp>

#include <boost/mpi.hpp>
#include <boost/bind.hpp>
#include <boost/serialization/map.hpp>

#include <string>

namespace alps {

    namespace detail {

        class params_impl_mpi : public params_impl_base, std::map<std::string, std::string> {

            public:

                typedef std::map<std::string, std::string> Base;

                params_impl_mpi(boost::mpi::communicator const & comm)
                    : comm_(comm)
                {}

                std::size_t size() const {
                    return Base::size();
                }

                std::vector<std::string> keys() const {
                    std::vector<std::string> arr;
                    for (Base::const_iterator it = Base::begin(); it != Base::end(); ++it)
                        arr.push_back(it->first);
                    return arr;
                }

                param operator[](std::string const & key) {
                    return param(
                        boost::bind(&params_impl_mpi::getter, boost::ref(*this), key),
                        boost::bind(&params_impl_mpi::setter, boost::ref(*this), key, _1)
                    );
                }

                param const operator[](std::string const & key) const {
                    if (!defined(key))
                        throw std::invalid_argument("unknown argument: "  + key + ALPS_STACKTRACE);
                    return param(Base::find(key)->second);
                }

                bool defined(std::string const & key) const {
                    return find(key) != end();
                }

                void save(hdf5::archive & ar) const {
                    ar << make_pvp("", static_cast<Base const &>(*this));
                }

                void load(hdf5::archive & ar) {
                    std::vector<std::string> list = ar.list_children(ar.get_context());
                    for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
                        std::string value;
                        ar >> make_pvp(*it, value);
                        insert(std::make_pair(*it, value));
                    }
                }
                
                params_impl_base * clone() {
                    return new params_impl_mpi(*this);
                }

                #ifdef ALPS_HAVE_MPI
                    void broadcast(int root) {
                        boost::mpi::broadcast(comm_, static_cast<Base &>(*this), root);
                    }
                #endif

            private:

                params_impl_mpi(params_impl_mpi const & arg)
                    : Base(arg)
                    , comm_(arg.comm_)
                {}

                void setter(std::string key, std::string value) {
                    Base::operator[](key) = value;
                }

                std::string getter(std::string key) {
                    return Base::operator[](key);
                }
                
                boost::mpi::communicator const & comm_;
        };

    }
}

#endif
