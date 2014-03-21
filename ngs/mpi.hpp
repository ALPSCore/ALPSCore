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

#ifndef ALPS_NGS_MPI_HPP
#define ALPS_NGS_MPI_HPP

#ifdef ALPS_HAVE_MPI

    #include <alps/hdf5.hpp>

    #include <alps/ngs/boost_mpi.hpp>

    namespace alps {
        namespace mpi {
            namespace detail {

                template<typename T, typename S> std::size_t copy_to_buffer(T const & values, std::vector<S> & buffer, std::size_t offset, boost::true_type) {
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> extent(get_extent(values));
                    std::size_t size = std::accumulate(extent.begin(), extent.end(), 0);
                    using alps::hdf5::get_pointer;
                    std::memcpy(&buffer[offset], const_cast<S *>(get_pointer(values)), sizeof(typename hdf5::scalar_type<T>::type) * size);
                    return size;
                }

                template<typename T, typename S> std::size_t copy_to_buffer(T const & values, std::vector<S> & buffer, std::size_t offset, boost::false_type) {
                    for(typename T::const_iterator it = values.begin(); it != values.end(); ++it)
                        offset += copy_to_buffer(*it, buffer, offset, typename hdf5::is_continuous<typename T::value_type>::type());
                    return offset;
                }

                template<typename T, typename S> std::size_t copy_from_buffer(T const & values, std::vector<S> & buffer, std::size_t offset, boost::true_type) {
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> extent(get_extent(values));
                    std::size_t size = std::accumulate(extent.begin(), extent.end(), 0);
                    using alps::hdf5::get_pointer;
                    std::memcpy(const_cast<S *>(get_pointer(values)), &buffer[offset], sizeof(typename hdf5::scalar_type<T>::type) * size);
                    return size;
                }

                template<typename T, typename S> std::size_t copy_from_buffer(T const & values, std::vector<S> & buffer, std::size_t offset, boost::false_type) {
                    for(typename T::const_iterator it = values.begin(); it != values.end(); ++it)
                        offset += copy_from_buffer(*it, buffer, offset, typename hdf5::is_continuous<typename T::value_type>::type());
                    return offset;
                }

                template<typename T, typename Op, typename C> void reduce_impl(const boost::mpi::communicator & comm, T const & in_values, Op op, int root, boost::true_type, C) {
                    using boost::mpi::reduce;
                    reduce(comm, in_values, op, root);
                }

                template<typename T, typename Op> void reduce_impl(const boost::mpi::communicator & comm, T const & in_values, Op op, int root, boost::false_type, boost::true_type) {
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> extent(get_extent(in_values));
                    using boost::mpi::reduce;
                    using alps::hdf5::get_pointer;
                    reduce(comm, get_pointer(in_values), std::accumulate(extent.begin(), extent.end(), 0), op, root);
                }

                template<typename T, typename Op, typename C> void reduce_impl(const boost::mpi::communicator & comm, T const & in_values, T & out_values, Op op, int root, boost::true_type, C) {
                    using boost::mpi::reduce;
                    reduce(comm, (T)in_values, out_values, op, root); // TODO: WTF? - why does boost not define unsigned long long as native datatype
                }

                template<typename T, typename Op> void reduce_impl(const boost::mpi::communicator & comm, T const & in_values, T & out_values, Op op, int root, boost::false_type, boost::true_type) {
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> extent(get_extent(in_values));
                    using alps::hdf5::set_extent;
                    set_extent(out_values, std::vector<std::size_t>(extent.begin(), extent.end()));
                    using boost::mpi::reduce;
                    using alps::hdf5::get_pointer;
                    reduce(comm, get_pointer(in_values), std::accumulate(extent.begin(), extent.end(), 0), get_pointer(out_values), op, root);
                }

                template<typename T, typename Op> void reduce_impl(const boost::mpi::communicator & comm, T const & in_values, Op op, int root, boost::false_type, boost::false_type) {
                    using alps::hdf5::is_vectorizable;
                    if (is_vectorizable(in_values)) {
                        using alps::hdf5::get_extent;
                        std::vector<std::size_t> extent(get_extent(in_values));
                        std::vector<typename alps::hdf5::scalar_type<T>::type> in_buffer(std::accumulate(extent.begin(), extent.end(), 0));
                        using detail::copy_to_buffer;
                        copy_to_buffer(in_values, in_buffer, 0, typename hdf5::is_content_continuous<T>::type());
                        using boost::mpi::reduce;
                        reduce(comm, &in_buffer.front(), in_buffer.size(), op, root);
                    } else
                        throw std::logic_error("No alps::mpi::reduce available for this type " + std::string(typeid(T).name()) + ALPS_STACKTRACE);
                }

                template<typename T, typename Op> void reduce_impl(const boost::mpi::communicator & comm, T const & in_values, T & out_values, Op op, int root, boost::false_type, boost::false_type) {
                    using alps::hdf5::is_vectorizable;
                    if (is_vectorizable(in_values)) {
                        using alps::hdf5::get_extent;
                        std::vector<std::size_t> extent(get_extent(in_values));
                        std::vector<typename alps::hdf5::scalar_type<T>::type> in_buffer(std::accumulate(extent.begin(), extent.end(), 1, std::multiplies<std::size_t>()));
                        std::vector<typename alps::hdf5::scalar_type<T>::type> out_buffer(in_buffer);
                        using detail::copy_to_buffer;
                        copy_to_buffer(in_values, in_buffer, 0, typename hdf5::is_content_continuous<T>::type());
                        using boost::mpi::reduce;
                        reduce(comm, &in_buffer.front(), in_buffer.size(), &out_buffer.front(), op, root);
                        using alps::hdf5::set_extent;
                        set_extent(out_values, std::vector<std::size_t>(extent.begin(), extent.end()));
                        using detail::copy_from_buffer;
                        copy_from_buffer(out_values, out_buffer, 0, typename hdf5::is_content_continuous<T>::type());
                    } else
                        throw std::logic_error("No alps::mpi::reduce available for this type " + std::string(typeid(T).name()) + ALPS_STACKTRACE);
                }
            }

            template<typename T, typename Op> void reduce(const boost::mpi::communicator & comm, T const & in_values, Op op, int root) {
                using detail::reduce_impl;
                reduce_impl(comm, in_values, op, root, typename boost::is_scalar<T>::type(), typename hdf5::is_content_continuous<T>::type());
            }

            template<typename T, typename Op> void reduce(const boost::mpi::communicator & comm, T const & in_values, T & out_values, Op op, int root) {
                using detail::reduce_impl;
                reduce_impl(comm, in_values, out_values, op, root, typename boost::is_scalar<T>::type(), typename hdf5::is_content_continuous<T>::type());
            }

        }
    }

#endif

#endif
