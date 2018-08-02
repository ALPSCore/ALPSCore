/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_TENSOR_HPP_H
#define ALPS_HDF5_TENSOR_HPP_H

#include <alps/hdf5/archive.hpp>
#include <alps/numeric/tensors/tensor_base.hpp>
#include <alps/type_traits/is_complex.hpp>


namespace alps {
  namespace hdf5 {
//
    template<typename T, std::size_t N, typename A> struct scalar_type<alps::numerics::detail::tensor_base<T, N, A> > {
      typedef typename scalar_type<typename std::remove_reference<typename std::remove_cv<T>::type>::type>::type type;
    };

    template<typename T, std::size_t N, typename A> struct has_complex_elements<alps::numerics::detail::tensor_base<T, N, A> >
      : public has_complex_elements<typename alps::detail::remove_cvr<T>::type>
    {};

    namespace detail {

      template<typename T, std::size_t N, typename A> struct get_extent<alps::numerics::detail::tensor_base<T, N, A> > {
        static std::vector<std::size_t> apply(const alps::numerics::detail::tensor_base<T, N, A> & value) {
          using alps::hdf5::get_extent;
          std::vector < std::size_t > result(value.shape().begin(), value.shape().begin() + N);
          if (value.size()) {
            std::vector < std::size_t > extent(get_extent(value.data()[0]));
            for (std::size_t i = 1; i < value.size(); ++i)
              if (!std::equal(extent.begin(), extent.end(), get_extent(value.data()[i]).begin()))
                throw archive_error("no rectangular matrix");
            std::copy(extent.begin(), extent.end(), std::back_inserter(result));
          }
          return result;
        }
      };

      template<typename T, std::size_t N, typename A> struct set_extent<alps::numerics::detail::tensor_base<T, N, A> > {
        static void apply(alps::numerics::detail::tensor_base<T, N, A> & value, std::vector<std::size_t> const & size) {
          using alps::hdf5::set_extent;
          using alps::hdf5::get_extent;
          if (N > size.size())
            throw archive_error("invalid data size");
          std::vector < std::size_t > extent(get_extent(value.data()[0]));
          std::array<size_t, N> new_size;
          std::copy(size.begin(), size.end() - extent.size(), new_size.begin());
          value.reshape(new_size);
        }
      };

      template<typename T, std::size_t N, typename A> struct get_pointer<alps::numerics::detail::tensor_base<T, N, A>> {
        static typename alps::hdf5::scalar_type<alps::numerics::detail::tensor_base<T, N, A>>::type * apply(alps::numerics::detail::tensor_base<T, N, A> & value) {
          using alps::hdf5::get_pointer;
          return get_pointer(value.storage().data(0));
        }
      };
        template<typename T, std::size_t N, typename A> struct get_pointer<const alps::numerics::detail::tensor_base<T, N, A> > {
        static typename alps::hdf5::scalar_type<alps::numerics::detail::tensor_base<T, N, A> >::type const * apply(alps::numerics::detail::tensor_base<T, N, A> const & value) {
          using alps::hdf5::get_pointer;
          return get_pointer(value.storage().data(0));
        }
      };

    }

    template<typename T, std::size_t N, typename S> void save(
      archive & ar
      , std::string const & path
      , const alps::numerics::detail::tensor_base<T, N, S>& value
      , std::vector<std::size_t> size = std::vector<std::size_t>()
      , std::vector<std::size_t> chunk = std::vector<std::size_t>()
      , std::vector<std::size_t> offset = std::vector<std::size_t>()
    ) {
      std::vector<std::size_t> extent = get_extent(value);
      std::copy(extent.begin(), extent.end(), std::back_inserter(size));
      std::copy(extent.begin(), extent.end(), std::back_inserter(chunk));
      std::fill_n(std::back_inserter(offset), extent.size(), 0);
      ar.write(path, get_pointer(value), size, chunk, offset);
    }

    template<typename T, std::size_t N, typename S> void load(
      archive & ar
      , std::string const & path
      , numerics::detail::tensor_base<T, N, S> & value
      , std::vector<std::size_t> chunk = std::vector<std::size_t>()
      , std::vector<std::size_t> offset = std::vector<std::size_t>()
    ) {
      if (ar.is_group(path))
        throw invalid_path("invalid path");
      else {
        if (ar.is_complex(path) != is_complex<T>::value)
          throw archive_error("no complex value in archive" + ALPS_STACKTRACE);
        std::vector<std::size_t> size(ar.extent(path));
        if (value.dimension() > size.size()) {
          throw wrong_dimensions("dimensions mismatched.");
        }
        set_extent(value, std::vector<std::size_t>(size.begin() + chunk.size(), size.end()));

        std::copy(size.begin() + chunk.size(), size.end(), std::back_inserter(chunk));
        std::fill_n(std::back_inserter(offset), size.size() - offset.size(), 0);
        ar.read(path, get_pointer(value), chunk, offset);
      }
    }

  }
}
#endif //ALPSCORE_TENSOR_HPP_H
