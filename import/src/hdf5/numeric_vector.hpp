/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2013 by Andreas Hehn <hehn@phys.ethz.ch>                          *
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

#ifndef ALPS_HDF5_NUMERIC_VECTOR_HPP
#define ALPS_HDF5_NUMERIC_VECTOR_HPP

#include <alps/hdf5/archive.hpp>
#include <alps/numeric/matrix/vector.hpp>

namespace alps {
namespace hdf5 {

        template <typename T, typename MemoryBlock>
        void save(
                  alps::hdf5::archive & ar
                  , std::string const & path
                  , alps::numeric::vector<T, MemoryBlock> const & value
                  , std::vector<std::size_t> size = std::vector<std::size_t>()
                  , std::vector<std::size_t> chunk = std::vector<std::size_t>()
                  , std::vector<std::size_t> offset = std::vector<std::size_t>()
                  ) {
            ar[path] << static_cast<MemoryBlock const&>(value);
        }
        template <typename T, typename MemoryBlock>
        void load(
                  alps::hdf5::archive & ar
                  , std::string const & path
                  , alps::numeric::vector<T, MemoryBlock> & value
                  , std::vector<std::size_t> chunk = std::vector<std::size_t>()
                  , std::vector<std::size_t> offset = std::vector<std::size_t>()
                  ) {
            MemoryBlock tmp;
            ar[path] >> tmp;
            value = alps::numeric::vector<T, MemoryBlock>(tmp.begin(), tmp.end());
        }
}
}
#endif // ALPS_HDF5_NUMERIC_VECTOR_HPP
