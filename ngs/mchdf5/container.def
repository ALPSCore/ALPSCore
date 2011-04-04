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

#ifndef ALPS_NGS_HDF5_VECTOR_TEMPLATE_ARGS
    #error "no template args defined"
#endif
#ifndef ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE
    #error "no template type defined"
#endif

#include <alps/ngs/convert.hpp>

#include <iterator>
#include <algorithm>

namespace alps {
	namespace hdf5 {

		template< ALPS_NGS_HDF5_VECTOR_TEMPLATE_ARGS > struct scalar_type< ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE > {
			typedef typename scalar_type<typename ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE ::value_type>::type type;
		};

		template< ALPS_NGS_HDF5_VECTOR_TEMPLATE_ARGS > struct has_complex_elements< ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE > 
			: public has_complex_elements<typename ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE ::value_type>
		{};

		namespace detail {

			template<ALPS_NGS_HDF5_VECTOR_TEMPLATE_ARGS> struct get_extent< ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE > {
				static std::vector<std::size_t> apply( ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE const & value) {
					using alps::hdf5::get_extent;
					std::vector<std::size_t> result(1, value.size());
					if (value.size()) {
						std::vector<std::size_t> first(get_extent(value[0]));
						for(typename ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE ::const_iterator it = value.begin() + 1; it != value.end(); ++it) {
							std::vector<std::size_t> size(get_extent(*it));
							if (
								   first.size() != size.size()
								|| !std::equal(first.begin(), first.end(), size.begin())
							)
								ALPS_NGS_THROW_RUNTIME_ERROR("no rectengual matrix")
						}
						std::copy(first.begin(), first.end(), std::back_inserter(result));
					}
					return result;
				}
			};

			template<ALPS_NGS_HDF5_VECTOR_TEMPLATE_ARGS> struct set_extent< ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE > {
				static void apply( ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE & value, std::vector<std::size_t> const & extent) {
					using alps::hdf5::set_extent;
					value.resize(extent[0]);
					if (extent.size() > 1)
						for(typename ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE ::iterator it = value.begin(); it != value.end(); ++it)
							set_extent(*it, std::vector<std::size_t>(extent.begin() + 1, extent.end()));
				}
			};

			template<ALPS_NGS_HDF5_VECTOR_TEMPLATE_ARGS> struct is_vectorizable< ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE > {
				static bool apply( ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE const & value) {
					using alps::hdf5::get_extent;
					using alps::hdf5::is_vectorizable;
					if (value.size()) {
						if (!is_vectorizable(value[0]))
							return false;
						std::vector<std::size_t> first(get_extent(value[0]));
						for(typename ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE ::const_iterator it = value.begin(); it != value.end(); ++it)
							if (!is_vectorizable(*it))
								return false;
							else {
								std::vector<std::size_t> size(get_extent(*it));
								if (
									   first.size() != size.size() 
									|| !std::equal(first.begin(), first.end(), size.begin())
								)
									return false;
							}
					}
					return true;
				}
			};

			template<ALPS_NGS_HDF5_VECTOR_TEMPLATE_ARGS> struct get_pointer< ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE > {
				static typename alps::hdf5::scalar_type<ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE>::type * apply( ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE & value) {
					using alps::hdf5::get_pointer;
					return get_pointer(value[0]);
				}
			};

			template<ALPS_NGS_HDF5_VECTOR_TEMPLATE_ARGS> struct get_pointer< ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE const> {
				static typename alps::hdf5::scalar_type<ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE>::type const * apply( ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE const & value) {
					using alps::hdf5::get_pointer;
					return get_pointer(value[0]);
				}
			};
		}

		template< ALPS_NGS_HDF5_VECTOR_TEMPLATE_ARGS > void save(
			  archive & ar
			, std::string const & path
			, ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE const & value
			, std::vector<std::size_t> size = std::vector<std::size_t>()
			, std::vector<std::size_t> chunk = std::vector<std::size_t>()
			, std::vector<std::size_t> offset = std::vector<std::size_t>()
		) {
			using alps::convert;
			if (ar.is_group(path))
				ar.delete_group(path);
			if (is_continous<T>::value && value.size() == 0)
				ar.write(path, static_cast<typename scalar_type< ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE >::type const *>(NULL), std::vector<std::size_t>());
			else if (is_continous<T>::value) {
				std::vector<std::size_t> extent(get_extent(value));
				std::copy(extent.begin(), extent.end(), std::back_inserter(size));
				std::copy(extent.begin(), extent.end(), std::back_inserter(chunk));
				std::fill_n(std::back_inserter(offset), extent.size(), 0);
				ar.write(path, get_pointer(value), size, chunk, offset);
			} else if (value.size() == 0)
				ar.write(path, static_cast<int const *>(NULL), std::vector<std::size_t>());
			else if (is_vectorizable(value)) {
				size.push_back(value.size());
				chunk.push_back(1);
				offset.push_back(0);
				for(typename ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE ::const_iterator it = value.begin(); it != value.end(); ++it) {
					offset.back() = it - value.begin();
					save(ar, path, *it, size, chunk, offset);
				}
			} else {
				if (ar.is_data(path))
					ar.delete_data(path);
				for(typename ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE ::const_iterator it = value.begin(); it != value.end(); ++it)
					save(ar, path + "/" + convert<std::string>(it - value.begin()), *it);
			}
		}

		template< ALPS_NGS_HDF5_VECTOR_TEMPLATE_ARGS > void load(
			  archive & ar
			, std::string const & path
			, ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE & value
			, std::vector<std::size_t> chunk = std::vector<std::size_t>()
			, std::vector<std::size_t> offset = std::vector<std::size_t>()
		) {
			using alps::convert;
			if (ar.is_group(path)) {
				std::vector<std::string> children = ar.list_children(path);
				value.resize(children.size());
				for (typename std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
					load(ar, path + "/" + *it, value[convert<std::size_t>(*it)]);
			} else {
				std::vector<std::size_t> size(ar.extent(path));
				if (is_continous<T>::value) {
					set_extent(value, std::vector<std::size_t>(size.begin() + chunk.size(), size.end()));
					if (value.size()) {
						std::copy(size.begin() + chunk.size(), size.end(), std::back_inserter(chunk));
						std::fill_n(std::back_inserter(offset), size.size() - offset.size(), 0);
						ar.read(path, get_pointer(value), chunk, offset);
					}
				} else {
					set_extent(value, std::vector<std::size_t>(1, *(size.begin() + chunk.size())));
					chunk.push_back(1);
					offset.push_back(0);
					for(typename ALPS_NGS_HDF5_VECTOR_TEMPLATE_TYPE ::iterator it = value.begin(); it != value.end(); ++it) {
						offset.back() = it - value.begin();
						load(ar, path, *it, chunk, offset);
					}
				}
			}
		}

	}
}
