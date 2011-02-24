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

#ifndef ALPS_NGS_HDF5_BOOST_NUMERIC_UBLAS_MATRIX_HPP
#define ALPS_NGS_HDF5_BOOST_NUMERIC_UBLAS_MATRIX_HPP

#include <alps/ngs/mchdf5.hpp>

#include <boost/numeric/ublas/matrix.hpp>

#include <iterator>

namespace alps {

    template <typename T, typename F, typename A> struct has_complex_elements< boost::numeric::ublas::matrix<T, F, A> >
        : public has_complex_elements<T>
    {};

    template <typename T, typename F, typename A> void serialize(
          mchdf5 & ar
        , std::string const & path
        , boost::numeric::ublas::matrix<T, F, A> const & value
        , std::vector<std::size_t> size = std::vector<std::size_t>()
        , std::vector<std::size_t> chunk = std::vector<std::size_t>()
        , std::vector<std::size_t> offset = std::vector<std::size_t>()
    ) {
        size.push_back(value.size1());
        size.push_back(value.size2());
        chunk.push_back(1);
        chunk.push_back(1);
        offset.push_back(0);
        offset.push_back(0);
        ar.write(path, &value(0, 0), size, chunk, offset);
    }

    template <typename T, typename F, typename A> void unserialize(
          mchdf5 & ar
        , std::string const & path
        , boost::numeric::ublas::matrix<T, F, A> & value
        , std::vector<std::size_t> chunk = std::vector<std::size_t>()
        , std::vector<std::size_t> offset = std::vector<std::size_t>()
    ) {
        if (is_continous<T>::value) {
            std::vector<std::size_t> size(ar.extent(path));
            value.resize(size[chunk.size()], size[chunk.size() + 1], false);
            std::copy(size.begin() + chunk.size(), size.end(), std::back_insert_iterator<std::vector<std::size_t> >(chunk));
            std::fill_n(std::back_insert_iterator<std::vector<std::size_t> >(offset), size.size() - offset.size(), 0);
            ar.read(path, &value(0, 0), chunk, offset);
        } else
            ALPS_NGS_THROW_RUNTIME_ERROR("invalid type")
    }

}

#endif
