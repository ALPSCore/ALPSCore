
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

#ifndef ALPS_NGS_HDF5_DEPRECATED
#define ALPS_NGS_HDF5_DEPRECATED

#include <alps/ngs/hdf5.hpp>

namespace alps {
    namespace hdf5 {

        class oarchive : public archive {
            public:
                oarchive(std::string const & file, bool compress = false) 
                    : archive(file, archive::WRITE || (compress ? archive::COMPRESS : 0x00))
                {}
                oarchive(oarchive const & ar)
                    : archive(ar)
                {}
        };

        class iarchive : public archive {
            public:
                iarchive(std::string const & file, bool compress = false) 
                    : archive(file, archive::READ || (compress ? archive::COMPRESS : 0x00))
                {}
                iarchive(oarchive const & ar)
                    : archive(ar)
                {}
        };

    }
}

#define ALPS_HDF5_HAVE_DEPRECATED

#include <alps/ngs/hdf5/map.hpp>
#include <alps/ngs/hdf5/pair.hpp>
#include <alps/ngs/hdf5/vector.hpp>
#include <alps/ngs/hdf5/pointer.hpp>
#include <alps/ngs/hdf5/complex.hpp>
#include <alps/ngs/hdf5/valarray.hpp>
#include <alps/ngs/hdf5/multi_array.hpp>
#include <alps/ngs/hdf5/shared_array.hpp>
#include <alps/ngs/hdf5/ublas/matrix.hpp>
#include <alps/ngs/hdf5/ublas/vector.hpp>

namespace alps {
    namespace hdf5 {

        template<typename T> void save(
               oarchive & ar
             , std::string const & path
             , T const & value
             , std::vector<std::size_t> size = std::vector<std::size_t>()
             , std::vector<std::size_t> chunk = std::vector<std::size_t>()
             , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (chunk.size())
                ALPS_NGS_THROW_RUNTIME_ERROR("user defined objects needs to be written continously");
            std::string context = ar.get_context();
            ar.set_context(ar.complete_path(path));
            value.serialize(ar);
            ar.set_context(context);
        }

        template<typename T> void load(
               iarchive & ar
             , std::string const & path
             , T & value
             , std::vector<std::size_t> chunk = std::vector<std::size_t>()
             , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (chunk.size())
                ALPS_NGS_THROW_RUNTIME_ERROR("user defined objects needs to be written continously");
            std::string context = ar.get_context();
            ar.set_context(ar.complete_path(path));
            value.serialize(ar);
            ar.set_context(context);
        }

        #define ALPS_NGS_HDF5_DEFINE_FREE_FUNCTIONS(T)                                                                                                                 \
            ALPS_DECL void save(                                                                                                                                                 \
                  oarchive & ar                                                                                                                                        \
                , std::string const & path                                                                                                                             \
                , T const & value                                                                                                                                      \
                , std::vector<std::size_t> size = std::vector<std::size_t>()                                                                                           \
                , std::vector<std::size_t> chunk = std::vector<std::size_t>()                                                                                          \
                , std::vector<std::size_t> offset = std::vector<std::size_t>()                                                                                         \
            );                                                                                                                                                         \
                                                                                                                                                                       \
            ALPS_DECL void load(                                                                                                                                                 \
                  iarchive & ar                                                                                                                                        \
                , std::string const & path                                                                                                                             \
                , T & value                                                                                                                                            \
                , std::vector<std::size_t> chunk = std::vector<std::size_t>()                                                                                          \
                , std::vector<std::size_t> offset = std::vector<std::size_t>()                                                                                         \
            );
        ALPS_NGS_FOREACH_NATIVE_HDF5_TYPE(ALPS_NGS_HDF5_DEFINE_FREE_FUNCTIONS)
        #undef ALPS_NGS_HDF5_DEFINE_FREE_FUNCTIONS

        template <typename T> typename boost::enable_if<
              has_complex_elements<typename detail::remove_cvr<T>::type>
            , oarchive &
        >::type operator<< (oarchive & ar, detail::make_pvp_proxy<T> const & proxy) {
            save(ar, proxy.path_, proxy.value_);
            ar.set_complex(proxy.path_);
            return ar;
        }

        template <typename T> typename boost::disable_if<
              has_complex_elements<typename detail::remove_cvr<T>::type>
            , oarchive &
        >::type operator<< (oarchive & ar, detail::make_pvp_proxy<T> const & proxy) {
            save(ar, proxy.path_, proxy.value_);
            return ar;
        }

        template <typename T> iarchive & operator>> (iarchive & ar, detail::make_pvp_proxy<T> proxy) {
            load(ar, proxy.path_, proxy.value_);
            return ar;
        }

    }
}

#endif
