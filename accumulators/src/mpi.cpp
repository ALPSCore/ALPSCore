/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/accumulators/mpi.hpp>
#include <alps/hdf5.hpp>

#ifdef ALPS_HAVE_MPI

    namespace alps {
        namespace alps_mpi {
            namespace detail {

                /// MPI_Reduce() with argument checking.
                /** @todo FIXME: Should be replaced with alps::mpi::reduce()
                                 once implemented properly, with error checking */
                inline int checked_mpi_reduce(const void* sendbuf, void* recvbuf, int count,
                                                     MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
                {
                    if (count<=0) {
                        throw std::invalid_argument("MPI_Reduce() is called with invalid count="
                                                    + std::to_string(count)
                                                    + ALPS_STACKTRACE);
                    }
                    if (sendbuf==recvbuf) {
                        throw std::invalid_argument("MPI_Reduce() is called with sendbuf==recvbuf"
                                                    + ALPS_STACKTRACE);
                    }
                    // WORKAROUND:
                    // for some reason, OpenMPI 1.6 declares `sendbuf` as `void*`, hence `const_cast`.
                    const int rc=MPI_Reduce(const_cast<void*>(sendbuf), recvbuf, count, datatype, op, root, comm);
                    return rc;
                }

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
                template<typename T, typename S> std::size_t copy_to_buffer(T const & values, std::vector<S> & buffer, std::size_t offset, std::true_type) {
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
                template<typename T, typename S> std::size_t copy_to_buffer(T const & values, std::vector<S> & buffer, std::size_t offset, std::false_type) {
                    /// FIXME!! BUG: if `T` is not vectorizable it may not have `begin()` and `end()` methods nor `const_iterator` type. This function won't be called --- but it gives compilation error!
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
                template<typename T, typename S> std::size_t copy_from_buffer(T const & values, std::vector<S> & buffer, std::size_t offset, std::true_type) {
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
                template<typename T, typename S> std::size_t copy_from_buffer(T const & values, std::vector<S> & buffer, std::size_t offset, std::false_type) {
                    for(typename T::const_iterator it = values.begin(); it != values.end(); ++it)
                        offset = copy_from_buffer(*it, buffer, offset, typename hdf5::is_continuous<typename T::value_type>::type());
                    return offset;
                }

                template<typename T, typename Op, typename C> void reduce_impl(const alps::mpi::communicator & comm, T const & in_values, Op /*op*/, int root, std::true_type, C) {
                    // using alps::mpi::reduce;
                    // reduce(comm, in_values, op, root);
                    using alps::mpi::get_mpi_datatype;
                    if (comm.rank()==root) {
                        throw std::logic_error("reduce_impl(): 4-arg overload is called by root rank."+ALPS_STACKTRACE);
                    }
                    checked_mpi_reduce((void*)&in_values, NULL, 1, get_mpi_datatype(T()),
                                       alps::mpi::is_mpi_op<Op, T>::op(), root, comm);

                }

                template<typename T, typename Op> void reduce_impl(const alps::mpi::communicator & comm, T const & in_values, Op /*op*/, int root, std::false_type, std::true_type) {
                    typedef typename alps::hdf5::scalar_type<T>::type scalar_type;
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> extent(get_extent(in_values));
                    using alps::hdf5::get_pointer;

                    // using boost::mpi::reduce;
                    // reduce(comm, get_pointer(in_values), std::accumulate(extent.begin(), extent.end(), 0), op, root);

                    using alps::mpi::get_mpi_datatype;
                    checked_mpi_reduce(const_cast<scalar_type*>(get_pointer(in_values)), NULL,
                                       std::accumulate(extent.begin(), extent.end(), 1, std::multiplies<std::size_t>()),
                                       get_mpi_datatype(scalar_type()), alps::mpi::is_mpi_op<Op, scalar_type>::op(), root, comm);
                }

                template<typename T, typename Op, typename C> void reduce_impl(const alps::mpi::communicator & comm, T const & in_values, T & out_values, Op /*op*/, int root, std::true_type, C) {
                    using alps::mpi::reduce;
                    // using boost::mpi::reduce;
                    // reduce(comm, (T)in_values, out_values, op, root); // TODO: WTF? - why does boost not define unsigned long long as native datatype
                    using alps::mpi::get_mpi_datatype;
                    // if (comm.rank()!=root) {
                        // // usleep((comm.rank()+1)*1000000); // DEBUG!
                        // std::cerr << "DEBUG:WARNING: rank=" << comm.rank() << " is not root=" << root
                        //           << " but called 5-argument reduce_impl()." + ALPS_STACKTRACE << std::endl;
                    // }
                    void* sendbuf=const_cast<T*>(&in_values);
                    if (sendbuf == &out_values) {
                        sendbuf=MPI_IN_PLACE;
                    }
                    checked_mpi_reduce(sendbuf, &out_values, 1, get_mpi_datatype(T()),
                                       alps::mpi::is_mpi_op<Op, T>::op(), root, comm);
                }

                template<typename T, typename Op> void reduce_impl(const alps::mpi::communicator & comm, T const & in_values, T & out_values, Op /*op*/, int root, std::false_type, std::true_type) {
                    typedef typename alps::hdf5::scalar_type<T>::type scalar_type;
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> extent(get_extent(in_values));
                    using alps::hdf5::set_extent;
                    set_extent(out_values, std::vector<std::size_t>(extent.begin(), extent.end()));
                    using alps::hdf5::get_pointer;

                    // using boost::mpi::reduce;
                    // reduce(comm, get_pointer(in_values), std::accumulate(extent.begin(), extent.end(), 0), get_pointer(out_values), op, root);

                    using alps::mpi::get_mpi_datatype;
                    checked_mpi_reduce(const_cast<scalar_type*>(get_pointer(in_values)), get_pointer(out_values),
                               std::accumulate(extent.begin(), extent.end(), 1, std::multiplies<std::size_t>()),
                               get_mpi_datatype(scalar_type()), alps::mpi::is_mpi_op<Op, scalar_type>::op(), root, comm);
                }

                template<typename T, typename Op> void reduce_impl(const alps::mpi::communicator & comm, T const & in_values, Op /*op*/, int root, std::false_type, std::false_type) {
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

                        using alps::mpi::get_mpi_datatype;
                        checked_mpi_reduce(&in_buffer.front(), NULL, in_buffer.size(), get_mpi_datatype(scalar_type()),
                                           alps::mpi::is_mpi_op<Op, scalar_type>::op(), root, comm);

                    } else
                        throw std::logic_error("No alps::mpi::reduce available for this type " + std::string(typeid(T).name()) + ALPS_STACKTRACE);
                }

                template<typename T, typename Op> void reduce_impl(const alps::mpi::communicator & comm, T const & in_values, T & out_values, Op /*op*/, int root, std::false_type, std::false_type) {
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

                        using alps::mpi::get_mpi_datatype;
                        checked_mpi_reduce(&in_buffer.front(), &out_buffer.front(), in_buffer.size(),
                                           get_mpi_datatype(scalar_type()),
                                           alps::mpi::is_mpi_op<Op, scalar_type>::op(), root, comm);

                        using alps::hdf5::set_extent;
                        set_extent(out_values, std::vector<std::size_t>(extent.begin(), extent.end()));
                        using detail::copy_from_buffer;
                        copy_from_buffer(out_values, out_buffer, 0, typename hdf5::is_content_continuous<T>::type());
                    } else
                        throw std::logic_error("No alps::mpi::reduce available for this type " + std::string(typeid(T).name()) + ALPS_STACKTRACE);
                }
            } // detail::

            template<typename T, typename Op> void reduce(const alps::mpi::communicator & comm, T const & in_values, Op op, int root) {
                using detail::reduce_impl;
                reduce_impl(comm, in_values, op, root, typename std::is_scalar<T>::type(), typename hdf5::is_content_continuous<T>::type());
            }

            template<typename T, typename Op> void reduce(const alps::mpi::communicator & comm, T const & in_values, T & out_values, Op op, int root) {
                using detail::reduce_impl;
                reduce_impl(comm, in_values, out_values, op, root, typename std::is_scalar<T>::type(), typename hdf5::is_content_continuous<T>::type());
            }

            using alps::mpi::communicator;

            #define ALPS_INST_MPI_REDUCE(T)                                                                                                       \
                template void reduce(const communicator &, T const &, std::plus<T>, int);                                                         \
                template void reduce(const communicator &, T const &, T &, std::plus<T>, int);                                                    \
                template void reduce(const communicator &, std::vector<T> const &, std::plus<T>, int);                                            \
                template void reduce(const communicator &, std::vector<T> const &, std::vector<T> &, std::plus<T>, int);                          \
                template void reduce(const communicator &, std::vector<std::vector<T>> const&, std::plus<T>, int);                                \
                template void reduce(const communicator &, std::vector<std::vector<T>> const&, std::vector<std::vector<T>>&, std::plus<T>, int);

            ALPS_INST_MPI_REDUCE(boost::uint64_t)
            ALPS_INST_MPI_REDUCE(float)
            ALPS_INST_MPI_REDUCE(double)
            ALPS_INST_MPI_REDUCE(long double)

        } // alps_mpi::
    } // alps::

#endif
