/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2012 by Andreas Hehn <hehn@phys.ethz.ch>                          *
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

#ifndef ALPS_NGS_HDF5_ALPS_MATRIX_HPP
#define ALPS_NGS_HDF5_ALPS_MATRIX_HPP

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/cast.hpp>
#include <alps/ngs/hdf5/complex.hpp>
#include <alps/numeric/matrix.hpp>

#include <algorithm>

// This one is only required for the old matrix hdf5 format
#include <alps/ngs/hdf5/vector.hpp>


namespace alps {
namespace hdf5 {

    template<typename T, typename MemoryBlock>
    struct scalar_type<alps::numeric::matrix<T, MemoryBlock> > {
        typedef typename scalar_type<typename alps::numeric::matrix<T,MemoryBlock>::value_type>::type type;
    };

    template<typename T, typename MemoryBlock>
    struct has_complex_elements<alps::numeric::matrix<T, MemoryBlock> >
    : public has_complex_elements<typename alps::detail::remove_cvr<typename alps::numeric::matrix<T, MemoryBlock>::value_type>::type>
    {};

namespace detail {

    //
    // Specializations of the alps::hdf5 helpers get_extent and set_extent
    //
    template <typename T, typename MemoryBlock>
    struct get_extent<alps::numeric::matrix<T,MemoryBlock> > {
        static std::vector<std::size_t> apply(alps::numeric::matrix<T,MemoryBlock> const& m) {
            using alps::hdf5::get_extent;
            using std::copy;
            using std::equal;
            typedef typename alps::numeric::matrix<T,MemoryBlock>::const_col_element_iterator col_iterator;
            std::vector<std::size_t> extent(2);
            extent[0] = num_cols(m);
            extent[1] = num_rows(m);
            if(num_rows(m) && num_cols(m) ) {
                // Get the extent of the first element
                std::vector<std::size_t> first(get_extent(m(0,0)));
                // We require that all elements of the matrix have the same size
                // These loops will check that.
                for(std::size_t j=0; j < num_cols(m); ++j)
                    for(std::pair<col_iterator,col_iterator> r = col(m,j); r.first != r.second; ++r.first) {
                        std::vector<std::size_t> size(get_extent(*r.first));
                        if(
                               first.size() != size.size()
                            || !equal(first.begin(),first.end(), size.begin())
                          )
                            throw archive_error("No rectengual matrix" + ALPS_STACKTRACE);
                    }
                copy(first.begin(),first.end(), std::back_inserter(extent));
            }
            return extent;
        }
    };

    template <typename T, typename MemoryBlock>
    struct set_extent<alps::numeric::matrix<T,MemoryBlock> > {
        static void apply(alps::numeric::matrix<T,MemoryBlock>& m, std::vector<std::size_t> const& size) {
            using alps::hdf5::set_extent;
            typedef typename alps::numeric::matrix<T,MemoryBlock>::const_col_element_iterator col_iterator;
            m.reserve(size[1],size[0]);
            m.resize(size[1],size[0]);
            assert(m.capacity().first  == size[1]);
            if( !is_continuous<T>::value && (size.size() != 2))
                for(std::size_t j=0; j < num_cols(m); ++j)
                    for(std::pair<col_iterator,col_iterator> r = col(m,j); r.first != r.second; ++r.first)
                        set_extent(*r.first, std::vector<std::size_t>(size.begin() + 2, size.end()));
                        // We assumed that all elements share the same extent information
        }
    };

    template <typename T, typename MemoryBlock>
    struct is_vectorizable<alps::numeric::matrix<T,MemoryBlock> > {
        static bool apply(alps::numeric::matrix<T,MemoryBlock> const& m) {
            typedef typename alps::numeric::matrix<T,MemoryBlock>::const_col_element_iterator col_iterator;
            using alps::hdf5::get_extent;
            using alps::hdf5::is_vectorizable;
            using std::equal;
            if(boost::is_scalar<typename alps::numeric::matrix<T,MemoryBlock>::value_type>::value || m.empty())
                return true;
            else {
                std::vector<std::size_t> first_element_extent(get_extent(m(0,0)));
                for(std::size_t j=0; j < num_cols(m); ++j) {
                    for(std::pair<col_iterator,col_iterator> r = col(m,j); r.first != r.second; ++r.first) {
                        if(!is_vectorizable(*r.first)) {
                            return false;
                        } else {
                            std::vector<std::size_t> element_extent(get_extent(*r.first));
                            if(
                                   first_element_extent.size() != element_extent.size()
                                || !equal(first_element_extent.begin(), first_element_extent.end(), element_extent.begin())
                            )
                                return false;
                        }
                    }
                }
                return true;
            }
        }
    };

    template <typename T, typename MemoryBlock>
    struct get_pointer<alps::numeric::matrix<T,MemoryBlock> > {
        static typename alps::hdf5::scalar_type<alps::numeric::matrix<T,MemoryBlock> >::type * apply(alps::numeric::matrix<T,MemoryBlock>& m) {
            using alps::hdf5::get_pointer;
            return get_pointer(m(0,0));
        }
    };

    template <typename T, typename MemoryBlock>
    struct get_pointer<alps::numeric::matrix<T,MemoryBlock> const> {
        static typename alps::hdf5::scalar_type<alps::numeric::matrix<T,MemoryBlock> >::type const * apply(alps::numeric::matrix<T,MemoryBlock> const& m) {
            using alps::hdf5::get_pointer;
            return get_pointer(m(0,0));
        }
    };

} // end namespace detail

    template <typename T, typename MemoryBlock> void save(
          archive& ar
        , std::string const& path
        , alps::numeric::matrix<T,MemoryBlock> const& m
        , std::vector<std::size_t> size   = std::vector<std::size_t>()
        , std::vector<std::size_t> chunk  = std::vector<std::size_t>()
        , std::vector<std::size_t> offset = std::vector<std::size_t>()
    ) {
        using std::copy;
        using std::fill_n;
        using alps::cast;
        typedef typename alps::numeric::matrix<T,MemoryBlock>::const_col_element_iterator col_iterator;
        if(is_continuous<T>::value) {
            std::vector<std::size_t> extent(get_extent(m));
            copy(extent.begin(),extent.end(), std::back_inserter(size));
            // We want to write one column:
            chunk.push_back(1);
            // How much memory does the column and the elements it contains need?
            copy(extent.begin()+1,extent.end(), std::back_inserter(chunk));
            std::size_t const offset_col_index = offset.size();
            fill_n(std::back_inserter(offset), extent.size(), 0);
            // Write column by column
            for(std::size_t j=0; j < num_cols(m); ++j) {
                offset[offset_col_index] = j;
                ar.write(path,get_pointer(*(col(m,j).first)),size,chunk,offset);
            }
        } else if( is_vectorizable(m) ){
            size.push_back(num_cols(m));
            size.push_back(num_rows(m));
            // We want to write element by element:
            chunk.push_back(1);
            chunk.push_back(1);
            std::size_t const offset_col_index = offset.size();
            std::size_t const offset_row_index = offset.size()+1;
            offset.push_back(0);
            offset.push_back(0);
            for(std::size_t j=0; j < num_cols(m); ++j) {
                offset[offset_col_index] = j;
                for(std::size_t i=0; i< num_rows(m); ++i) {
                    offset[offset_row_index] = i;
                    save(ar, path, m(i,j), size, chunk, offset);
                }
            }
        } else {
            if( ar.is_data(path) )
                ar.delete_data(path);
            for(std::size_t j=0; j < num_cols(m); ++j)
                for(std::size_t i=0; i < num_rows(m); ++i)
                    save(ar, ar.complete_path(path) + "/" + cast<std::string>(j) + "/" + cast<std::string>(i), m(i,j) );
        }
    }

    template <typename T, typename MemoryBlock> void load(
          archive & ar
        , std::string const & path
        , alps::numeric::matrix<T,MemoryBlock> & m
        , std::vector<std::size_t> chunk  = std::vector<std::size_t>()
        , std::vector<std::size_t> offset = std::vector<std::size_t>()
    ){
        using std::copy;
        if(ar.is_data(path + "/size1") && ar.is_scalar(path + "/size1") && ar.is_datatype<std::size_t>(path + "/size1")) {
            // Old matrix hdf5 format
            std::size_t size1(0), size2(0), reserved_size1(0);
            ar[path + "/size1"] >> size1;
            ar[path + "/size2"] >> size2;
            ar[path + "/reserved_size1"] >> reserved_size1;
            std::vector<T> data;
            ar[path + "/values"] >> data;
            alps::numeric::matrix<T,MemoryBlock> m2(reserved_size1,size2);
            assert(m2.capacity().first  == reserved_size1);
            copy(data.begin(), data.end(), col(m2,0).first);
            m2.resize(size1,size2);
            swap(m,m2);
            return;
        }

        alps::numeric::matrix<T,MemoryBlock> m2;
        if(ar.is_group(path)) {
            std::vector<std::string> const columns = ar.list_children(path);
            std::vector<std::string> const rows = ar.list_children(path + "/" +columns[0]);
            m2.resize(rows.size(), columns.size());
            // Check if all columns have the same number of elements
            for(std::vector<std::string>::const_iterator it = columns.begin(); it != columns.end(); ++it) {
                std::vector<std::string> const elements = ar.list_children(path + "/" + *it);
                if(elements.size() != rows.size())
                    throw invalid_path("invalid path" + ALPS_STACKTRACE);
                for(std::vector<std::string>::const_iterator eit = elements.begin(); eit != elements.end(); ++eit) {
                    load(ar, ar.complete_path(path) + "/" + *it + "/" + *eit, m2(cast<std::size_t>(*eit),cast<std::size_t>(*it)));
                }
            }
        } else {
            if (ar.is_complex(path) != has_complex_elements<T>::value)
                throw archive_error("no complex value in archive" + ALPS_STACKTRACE);
            std::vector<std::size_t> size(ar.extent(path));
            if(is_continuous<T>::value) {
                // We need to make sure that reserve() will reserve exactly the num_rows() we asked for.
                // The only way to ensure that is by creating a new matrix which has no reserved space.
                set_extent(m2,std::vector<std::size_t>(size.begin() + chunk.size(), size.end()));
                copy(size.begin()+chunk.size(), size.end(), std::back_inserter(chunk));
                fill_n(std::back_inserter(offset), size.size() - offset.size(), 0);
                ar.read(path,get_pointer(m2(0,0)),chunk,offset);
            } else { // i.e. is_vectorizable
                std::vector<std::size_t> mysize;
                mysize.push_back(*(size.begin() + chunk.size()));
                mysize.push_back(*(size.begin() + chunk.size()+1));
                set_extent(m2, mysize);
                // Read in the matrix element by element
                chunk.push_back(1);
                chunk.push_back(1);
                std::size_t const offset_col_index = offset.size();
                std::size_t const offset_row_index = offset.size()+1;
                offset.push_back(0);
                offset.push_back(0);
                for(std::size_t j=0; j < num_cols(m2); ++j) {
                    offset[offset_col_index] = j;
                    for(std::size_t i=0; i < num_rows(m2); ++i) {
                        offset[offset_row_index] = i;
                        load(ar, path, m2(i,j), chunk, offset);
                    }
                }
            }
        }
        swap(m,m2);
    }
} // end namespace hdf5
} // end namespace alps

#endif //ALPS_NGS_HDF5_ALPS_MATRIX_HPP
