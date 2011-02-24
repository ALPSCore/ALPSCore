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

#ifndef ALPS_NGS_HDF5_STD_MAP
#define ALPS_NGS_HDF5_STD_MAP

#include <alps/ngs/mchdf5.hpp>
#include <alps/ngs/convert.hpp>

#include <map>

namespace alps {

    template <typename K, typename T, typename C, typename A> void serialize(
          mchdf5 & ar
        , std::string const & path
        , std::map<K, T, C, A> const & value
        , std::vector<std::size_t> size = std::vector<std::size_t>()
        , std::vector<std::size_t> chunk = std::vector<std::size_t>()
        , std::vector<std::size_t> offset = std::vector<std::size_t>()
    ) {
        for(typename std::map<K, T, C, A>::const_iterator it = value.begin(); it != value.end(); ++it)
            serialize(ar, path + "/" + convert<std::string>(it->first), it->second);
    }

    template <typename K, typename T, typename C, typename A> void unserialize(
          mchdf5 & ar
        , std::string const & path
        , std::map<K, T, C, A> & value
        , std::vector<std::size_t> chunk = std::vector<std::size_t>()
        , std::vector<std::size_t> offset = std::vector<std::size_t>()
    ) {
        std::vector<std::string> children = ar.list_children(path);
        for (typename std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
            unserialize(ar, path + "/" + *it, value[convert<std::size_t>(*it)]);
    }

}

#endif
