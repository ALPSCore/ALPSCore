/*****************************************************************************
 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations
 *
 * ALPS Libraries
 *
 * Copyright (C) 2008-2018 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
 *                            Lukas Gamper <gamperl -at- gmail.com>
 *
 * This software is part of the ALPS libraries, published under the ALPS
 * Library License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 * 
 * You should have received a copy of the ALPS Library License along with
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef ALPS_HDF5_SHARED_PTR_HPP
#define ALPS_HDF5_SHARED_PTR_HPP

#include <alps/hdf5.hpp>

namespace alps {

#define ALPS_HDF5_MAKE_PVP(ptr_type, arg_type)                                                                                                          \
        template <typename T> hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > > make_pvp(std::string const & p, arg_type v, std::size_t s) {       \
            return hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > >(p, std::make_pair(&*v, std::vector<std::size_t>(1, s)));                      \
        }                                                                                                                                                   \
        template <typename T> hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > > make_pvp(                                                          \
            std::string const & p, arg_type v, std::vector<std::size_t> const & s                                                                           \
        ) {                                                                                                                                                 \
            return hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > >(p, std::make_pair(&*v, s));                                                   \
        }
    ALPS_HDF5_MAKE_PVP(T *, boost::shared_ptr<T> &)
    ALPS_HDF5_MAKE_PVP(T const *, boost::shared_ptr<T> const &)
    ALPS_HDF5_MAKE_PVP(T *, std::auto_ptr<T> &)
    ALPS_HDF5_MAKE_PVP(T const *, std::auto_ptr<T> const &)
    ALPS_HDF5_MAKE_PVP(T *, boost::weak_ptr<T> &)
    ALPS_HDF5_MAKE_PVP(T const *, boost::weak_ptr<T> const &)
    ALPS_HDF5_MAKE_PVP(T *, boost::scoped_ptr<T> &)
    ALPS_HDF5_MAKE_PVP(T const *, boost::scoped_ptr<T> const &)
    #undef ALPS_HDF5_MAKE_PVP

    #define ALPS_HDF5_MAKE_ARRAY_PVP(ptr_type, arg_type)                                                                                                    \
        template <typename T> hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > > make_pvp(std::string const & p, arg_type v, std::size_t s) {       \
            return hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > >(p, std::make_pair(v.get(), std::vector<std::size_t>(1, s)));                  \
        }                                                                                                                                                   \
        template <typename T> hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > > make_pvp(                                                          \
            std::string const & p, arg_type v, std::vector<std::size_t> const & s                                                                           \
        ) {                                                                                                                                                 \
            return hdf5::pvp<std::pair<ptr_type, std::vector<std::size_t> > >(p, std::make_pair(v.get(), s));                                               \
        }
    ALPS_HDF5_MAKE_ARRAY_PVP(T *, boost::shared_array<T> &)
    ALPS_HDF5_MAKE_ARRAY_PVP(T const *, boost::shared_array<T> const &)
    #undef ALPS_HDF5_MAKE_ARRAY_PVP
}
#endif
