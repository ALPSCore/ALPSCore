/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_ACCUMULATOR_MPI_HPP
#define ALPS_ACCUMULATOR_MPI_HPP

#include <alps/config.hpp>

#ifdef ALPS_HAVE_MPI

    #include <alps/hdf5.hpp>

    #include <alps/utilities/boost_mpi.hpp>

    #include <cassert>

    namespace alps {
        namespace mpi {
            namespace detail {

                /** @brief Copy a continous-type value (array of values) into a buffer.

                    The value is of type T representable as an array
                    of values of type S. The values are copied from `values` to `buffer[offset]`.

                    @warning FIXME (design bug?) The buffer (which is
                    a vector of type S) is accessed via a direct
                    memory copy and is assumed to have sufficient
                    size.
                    
                    @param values: value to copy from;
                    @param buffer: vector to copy to, starting from offset;
                    @param offset: position in the buffer to copy to;
                    @returns the new offset pointing right after the last used position in the buffer.
                 */
                template<typename T, typename S> std::size_t copy_to_buffer(T const & values, std::vector<S> & buffer, std::size_t offset, boost::true_type) {
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> extent(get_extent(values));
                    std::size_t size = std::accumulate(extent.begin(), extent.end(), 1, std::multiplies<std::size_t>());
                    assert(buffer.size()>=offset+size && "buffer has sufficient size to accommodate values");
                    using alps::hdf5::get_pointer;
                    std::memcpy(&buffer[offset], const_cast<S *>(get_pointer(values)), sizeof(typename hdf5::scalar_type<T>::type) * size);
                    return offset+size;
                }

                /** @brief Copy a container value into a buffer.

                    The container value is of type T which ultimately
                    holds values of a scalar type S (T can, e.g., be a
                    container of containers of containers... of
                    S). The values are copied from `values` to
                    `buffer[offset]`.

                    @param values: container to copy from;
                    @param buffer: vector to copy to, starting from offset;
                    @param offset: position in the buffer to copy to;
                    @returns the new offset pointing right after the last used position in the buffer.
                 */
                template<typename T, typename S> std::size_t copy_to_buffer(T const & values, std::vector<S> & buffer, std::size_t offset, boost::false_type) {
                    for(typename T::const_iterator it = values.begin(); it != values.end(); ++it)
                        offset = copy_to_buffer(*it, buffer, offset, typename hdf5::is_continuous<typename T::value_type>::type());
                    return offset;
                }

                /** @brief Copy a buffer into a continous-type value (array of values).

                    The value is of type T representable as an array
                    of values of type S. The values are copied from
                    `buffer[offset]` to `values`.

                    @warning FIXME (design bug?) The buffer (which is
                    a vector of type S) is accessed via a direct
                    memory copy and is assumed to have sufficient
                    size.
                    
                    @warning FIXME (design bug?) Although the
                    container `values` is declared as a const
                    reference, its contents is modified by this
                    function.

                    @param values: value to copy to;
                    @param buffer: vector to copy from, starting from offset;
                    @param offset: position in the buffer to copy from;
                    @returns the new offset pointing right after the last used position in the buffer.
                 */
                template<typename T, typename S> std::size_t copy_from_buffer(T const & values, std::vector<S> & buffer, std::size_t offset, boost::true_type) {
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> extent(get_extent(values));
                    std::size_t size = std::accumulate(extent.begin(), extent.end(), 1, std::multiplies<std::size_t>());
                    using alps::hdf5::get_pointer;
                    std::memcpy(const_cast<S *>(get_pointer(values)), &buffer[offset], sizeof(typename hdf5::scalar_type<T>::type) * size);
                    return offset+size;
                }

                /** @brief Copy a buffer into a container value.

                    The container value is of type T which ultimately
                    holds values of a scalar type S (T can, e.g., be a
                    container of containers of containers... of
                    S). The values are copied from `buffer[offset]` to `values`.

                    @param values: container to copy to;
                    @param buffer: vector to copy from, starting from offset;
                    @param offset: position in the buffer to copy from;
                    @returns the new offset pointing right after the last used position in the buffer.

                    @warning FIXME (design bug?) Although the
                    container `values` is declared as a const
                    reference, its content is modified by this
                    function.
                 */
                template<typename T, typename S> std::size_t copy_from_buffer(T const & values, std::vector<S> & buffer, std::size_t offset, boost::false_type) {
                    for(typename T::const_iterator it = values.begin(); it != values.end(); ++it)
                        offset = copy_from_buffer(*it, buffer, offset, typename hdf5::is_continuous<typename T::value_type>::type());
                    return offset;
                }

                template<typename T, typename Op, typename C> void reduce_impl(const boost::mpi::communicator & comm, T const & in_values, Op op, int root, boost::true_type, C) {
                    using boost::mpi::reduce;
                    reduce(comm, in_values, op, root);
                }

                template<typename T, typename Op> void reduce_impl(const boost::mpi::communicator & comm, T const & in_values, Op op, int root, boost::false_type, boost::true_type) {
                    typedef typename alps::hdf5::scalar_type<T>::type scalar_type;
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> extent(get_extent(in_values));
                    using alps::hdf5::get_pointer;

                    // using boost::mpi::reduce;
                    // reduce(comm, get_pointer(in_values), std::accumulate(extent.begin(), extent.end(), 0), op, root);

                    using boost::mpi::get_mpi_datatype;
                    MPI_Reduce(const_cast<scalar_type*>(get_pointer(in_values)), NULL, std::accumulate(extent.begin(), extent.end(), 1, std::multiplies<std::size_t>()), get_mpi_datatype(scalar_type()), boost::mpi::is_mpi_op<Op, scalar_type>::op(), root, comm);
                }

                template<typename T, typename Op, typename C> void reduce_impl(const boost::mpi::communicator & comm, T const & in_values, T & out_values, Op op, int root, boost::true_type, C) {
                    using boost::mpi::reduce;
                    reduce(comm, (T)in_values, out_values, op, root); // TODO: WTF? - why does boost not define unsigned long long as native datatype
                }

                template<typename T, typename Op> void reduce_impl(const boost::mpi::communicator & comm, T const & in_values, T & out_values, Op op, int root, boost::false_type, boost::true_type) {
                    typedef typename alps::hdf5::scalar_type<T>::type scalar_type;
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> extent(get_extent(in_values));
                    using alps::hdf5::set_extent;
                    set_extent(out_values, std::vector<std::size_t>(extent.begin(), extent.end()));
                    using alps::hdf5::get_pointer;

                    // using boost::mpi::reduce;
                    // reduce(comm, get_pointer(in_values), std::accumulate(extent.begin(), extent.end(), 0), get_pointer(out_values), op, root);

                    using boost::mpi::get_mpi_datatype;
                    MPI_Reduce(const_cast<scalar_type*>(get_pointer(in_values)), get_pointer(out_values), std::accumulate(extent.begin(), extent.end(), 1, std::multiplies<std::size_t>()), get_mpi_datatype(scalar_type()), boost::mpi::is_mpi_op<Op, scalar_type>::op(), root, comm);
                }

                template<typename T, typename Op> void reduce_impl(const boost::mpi::communicator & comm, T const & in_values, Op op, int root, boost::false_type, boost::false_type) {
                    using alps::hdf5::is_vectorizable;
                    if (is_vectorizable(in_values)) {
                        using alps::hdf5::get_extent;
                        typedef typename alps::hdf5::scalar_type<T>::type scalar_type;
                        std::vector<std::size_t> extent(get_extent(in_values));
			std::vector<scalar_type> in_buffer(std::accumulate(extent.begin(), extent.end(), 1, std::multiplies<std::size_t>()));
                        using detail::copy_to_buffer;
                        copy_to_buffer(in_values, in_buffer, 0, typename hdf5::is_content_continuous<T>::type());

                        // using boost::mpi::reduce;
                        // reduce(comm, &in_buffer.front(), in_buffer.size(), op, root);

                        using boost::mpi::get_mpi_datatype;
                        MPI_Reduce(&in_buffer.front(), NULL, in_buffer.size(), get_mpi_datatype(scalar_type()), boost::mpi::is_mpi_op<Op, scalar_type>::op(), root, comm);

                    } else
                        throw std::logic_error("No alps::mpi::reduce available for this type " + std::string(typeid(T).name()) + ALPS_STACKTRACE);
                }

                template<typename T, typename Op> void reduce_impl(const boost::mpi::communicator & comm, T const & in_values, T & out_values, Op op, int root, boost::false_type, boost::false_type) {
                    using alps::hdf5::is_vectorizable;
                    if (is_vectorizable(in_values)) {
                        using alps::hdf5::get_extent;
                        typedef typename alps::hdf5::scalar_type<T>::type scalar_type;
                        std::vector<std::size_t> extent(get_extent(in_values));
                        std::vector<scalar_type> in_buffer(std::accumulate(extent.begin(), extent.end(), 1, std::multiplies<std::size_t>()));
                        std::vector<scalar_type> out_buffer(in_buffer);
                        using detail::copy_to_buffer;
                        copy_to_buffer(in_values, in_buffer, 0, typename hdf5::is_content_continuous<T>::type());

                        // using boost::mpi::reduce;
                        // reduce(comm, &in_buffer.front(), in_buffer.size(), &out_buffer.front(), op, root);

                        using boost::mpi::get_mpi_datatype;
                        MPI_Reduce(&in_buffer.front(), &out_buffer.front(), in_buffer.size(), get_mpi_datatype(scalar_type()), boost::mpi::is_mpi_op<Op, scalar_type>::op(), root, comm);

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
