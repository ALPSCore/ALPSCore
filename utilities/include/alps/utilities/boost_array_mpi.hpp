/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>

#ifdef ALPS_HAVE_MPI

#include <boost/mpi.hpp>
#include <boost/array.hpp>
#include <boost/multi_array.hpp>

    namespace boost {
        namespace mpi {
          // boost::array
          template<typename T, std::size_t N, typename Op> void reduce(const communicator & comm, boost::array<T, N> const & in_values, Op op, int root) {
            reduce(comm, &in_values.front(), in_values.size(), op, root);
          }


          template<typename T, std::size_t N, typename Op> void reduce(const communicator & comm, boost::array<T, N> const & in_values, boost::array<T, N> & out_values, Op op, int root) {
            reduce(comm, &in_values.front(), in_values.size(), &out_values.front(), op, root);
          }

          // boost::mulit_array
          template<typename T, std::size_t N, typename A, typename Op> void reduce(const communicator & comm, boost::multi_array<T, N, A> const & in_values, Op op, int root) {
            reduce(comm, in_values.data(), std::accumulate(in_values.shape(), in_values.shape() + boost::multi_array<T, N, A>::dimensionality, 0), op, root);
          }

          template<typename T, std::size_t N, typename A, typename Op> void reduce(const communicator & comm, boost::multi_array<T, N, A> const & in_values, boost::multi_array<T, N, A> & out_values, Op op, int root) {
            boost::array<typename boost::multi_array<T, N, A>::size_type, boost::multi_array<T, N, A>::dimensionality> shape;
            std::copy(in_values.shape(), in_values.shape() + boost::multi_array<T, N, A>::dimensionality, shape.begin());
            out_values.resize(shape);
          reduce(comm, in_values.data(), std::accumulate(shape.begin(), shape.end(), 0), out_values.data(), op, root);
          }

        }
    }

#endif

