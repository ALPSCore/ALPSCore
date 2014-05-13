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

#ifndef ALPS_NGS_HDF5_HPP
#define ALPS_NGS_HDF5_HPP

#include <alps/ngs/config.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/hdf5/errors.hpp>
#include <alps/ngs/detail/remove_cvr.hpp>
#include <alps/ngs/detail/type_wrapper.hpp>

#include <boost/mpl/and.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_array.hpp>
#include <boost/type_traits/remove_all_extents.hpp>

#ifndef ALPS_NGS_SINGLE_THREAD

#include <boost/thread.hpp>

#endif

#include <map>
#include <vector>
#include <string>
#include <numeric>

#define ALPS_NGS_FOREACH_NATIVE_HDF5_TYPE(CALLBACK)                                                                                            \
    CALLBACK(char)                                                                                                                             \
    CALLBACK(signed char)                                                                                                                      \
    CALLBACK(unsigned char)                                                                                                                    \
    CALLBACK(short)                                                                                                                            \
    CALLBACK(unsigned short)                                                                                                                   \
    CALLBACK(int)                                                                                                                              \
    CALLBACK(unsigned)                                                                                                                         \
    CALLBACK(long)                                                                                                                             \
    CALLBACK(unsigned long)                                                                                                                    \
    CALLBACK(long long)                                                                                                                        \
    CALLBACK(unsigned long long)                                                                                                               \
    CALLBACK(float)                                                                                                                            \
    CALLBACK(double)                                                                                                                           \
    CALLBACK(long double)                                                                                                                      \
    CALLBACK(bool)                                                                                                                             \
    CALLBACK(std::string)

namespace alps {
    namespace hdf5 {

        namespace detail {
            struct archivecontext;

            template<typename A, typename T> struct is_datatype_caller {
                static bool apply(A const & ar, std::string path) {
                    throw std::logic_error("only native datatypes can be probed: " + path + ALPS_STACKTRACE);
                    return false;
                }
            };

            #define ALPS_NGS_HDF5_IS_DATATYPE_CALLER(T)                                                                                                                \
                template<typename A> struct is_datatype_caller<A, T > {                                                                                                \
                    static bool apply(A const & ar, std::string path, T unused = alps::detail::type_wrapper<T>::type()) {                                              \
                        return ar.is_datatype_impl(path, unused);                                                                                                      \
                    }                                                                                                                                                  \
                };
            ALPS_NGS_FOREACH_NATIVE_HDF5_TYPE(ALPS_NGS_HDF5_IS_DATATYPE_CALLER)
            #undef ALPS_NGS_HDF5_IS_DATATYPE_CALLER

            template<typename A> struct archive_proxy {

                explicit archive_proxy(std::string const & path, A & ar)
                    : path_(path), ar_(ar)
                {}

                template<typename T> archive_proxy & operator=(T const & value);
                template<typename T> archive_proxy & operator<<(T const & value);
                template <typename T> archive_proxy & operator>>(T & value);

                std::string path_;
                A ar_;
            };
        }

        class ALPS_DECL archive {

            public:

                // TODO: make this private
                typedef enum {
                    READ = 0x00, 
                    WRITE = 0x01, 
                    REPLACE = 0x02, 
                    COMPRESS = 0x04, 
                    LARGE = 0x08, 
                    MEMORY = 0x10 
                } properties;

                archive(boost::filesystem::path const & filename, std::string mode = "r");
                explicit archive(std::string const & filename, int props); // TODO: remove that!
                explicit archive(std::string const & filename, char prop); // TODO: remove that!
                explicit archive(std::string const & filename, char signed prop); // TODO: remove that!
                archive(archive const & arg);

                virtual ~archive();
                static void abort();

                std::string const & get_filename() const;

                std::string encode_segment(std::string segment) const;
                std::string decode_segment(std::string segment) const;

                std::string get_context() const;
                void set_context(std::string const & context);
                std::string complete_path(std::string path) const;

                void close();
                bool is_open();

                bool is_data(std::string path) const;
                bool is_attribute(std::string path) const;
                bool is_group(std::string path) const;

                bool is_scalar(std::string path) const;
                bool is_null(std::string path) const;
                bool is_complex(std::string path) const;

                template<typename T> bool is_datatype(std::string path) const {
                    return detail::is_datatype_caller<archive, T>::apply(*this, path);
                }

                std::vector<std::string> list_children(std::string path) const;
                std::vector<std::string> list_attributes(std::string path) const;

                std::vector<std::size_t> extent(std::string path) const;
                std::size_t dimensions(std::string path) const;

                void create_group(std::string path) const;

                void delete_data(std::string path) const;
                void delete_group(std::string path) const;
                void delete_attribute(std::string path) const;

                void set_complex(std::string path);

/* TODO: implement
                void move_data(std::string current_path, std::string new_path) const;
                void move_attribute(std::string current_path, std::string new_path) const;
*/

                detail::archive_proxy<archive> operator[](std::string const & path);

                template<typename T> void read(
                      std::string path
                    , T *
                    , std::vector<std::size_t>
                    , std::vector<std::size_t> = std::vector<std::size_t>()
                ) const {
                    throw std::logic_error("Invalid type on path: " + path + ALPS_STACKTRACE);
                }

                template<typename T> void write(
                      std::string path
                    , T const * value
                    , std::vector<std::size_t> size
                    , std::vector<std::size_t> chunk = std::vector<std::size_t>()
                    , std::vector<std::size_t> offset = std::vector<std::size_t>()
                ) const {
                    throw std::logic_error("Invalid type on path: " + path + ALPS_STACKTRACE);
                }

                #define ALPS_NGS_HDF5_DEFINE_API(T)                                                                                                                    \
                    void read(std::string path, T & value) const;                                                                                                      \
                    void read(                                                                                                                                         \
                          std::string path                                                                                                                             \
                        , T * value                                                                                                                                    \
                        , std::vector<std::size_t> chunk                                                                                                               \
                        , std::vector<std::size_t> offset = std::vector<std::size_t>()                                                                                 \
                    ) const;                                                                                                                                           \
                                                                                                                                                                       \
                    void write(std::string path, T value) const;                                                                                                       \
                    void write(                                                                                                                                        \
                          std::string path                                                                                                                             \
                        , T const * value, std::vector<std::size_t> size                                                                                               \
                        , std::vector<std::size_t> chunk = std::vector<std::size_t>()                                                                                  \
                        , std::vector<std::size_t> offset = std::vector<std::size_t>()                                                                                 \
                    ) const;
                ALPS_NGS_FOREACH_NATIVE_HDF5_TYPE(ALPS_NGS_HDF5_DEFINE_API)
                #undef ALPS_NGS_HDF5_DEFINE_API

                #define ALPS_NGS_HDF5_IS_DATATYPE_IMPL_DECL(T)                                                                                                         \
                    bool is_datatype_impl(std::string path, T) const;
                ALPS_NGS_FOREACH_NATIVE_HDF5_TYPE(ALPS_NGS_HDF5_IS_DATATYPE_IMPL_DECL)
                #undef ALPS_NGS_HDF5_IS_DATATYPE_IMPL_DECL

            private:

                void construct(std::string const & filename, std::size_t props = READ);
                std::string file_key(std::string filename, bool large, bool memory) const;

                std::string current_;
                detail::archivecontext * context_;

#ifndef ALPS_NGS_SINGLE_THREAD
                static boost::recursive_mutex mutex_;
#endif
                static std::map<std::string, std::pair<detail::archivecontext *, std::size_t> > ref_cnt_;

        };

        template<typename T> struct is_continuous
            : public boost::false_type
        {};

        template<typename T> struct is_content_continuous
            : public is_continuous<T>
        {};

        template<typename T> struct has_complex_elements
            : public boost::false_type
        {};
        
        template<typename T> struct scalar_type {
            typedef T type;
        };

        namespace detail {

             template<typename T> struct get_extent {
                static std::vector<std::size_t> apply(T const & value) {
                    return std::vector<std::size_t>();
                }
            };

            template<typename T> struct set_extent {
                 static void apply(T &, std::vector<std::size_t> const &) {}
            };
            
            #define ALPS_NGS_HDF5_DEFINE_SET_EXTENT(T)                                                                                                              \
                template<> struct set_extent<T> {                                                                                                                   \
                    static void apply(T &, std::vector<std::size_t> const & extent) {                                                                               \
                        if (extent.size() > 0)                                                                                                                      \
                            throw wrong_type("The extents do not match" + ALPS_STACKTRACE);                                                                         \
                    }                                                                                                                                               \
                };
            ALPS_NGS_FOREACH_NATIVE_HDF5_TYPE(ALPS_NGS_HDF5_DEFINE_SET_EXTENT)
            #undef ALPS_NGS_HDF5_DEFINE_SET_EXTENT

            template<typename T> struct is_vectorizable {
                 static bool apply(T const & value){
                    return false;
                }
            };

            template<typename T> struct get_pointer {
                 static typename alps::hdf5::scalar_type<T>::type * apply(T &) {
                    return NULL;
                }
            };

            template<typename T> struct get_pointer<T const> {
                 static typename alps::hdf5::scalar_type<T>::type const * apply(T const &) {
                    return NULL;
                }
            };

        }

        template<typename T> typename scalar_type<T>::type * get_pointer(T & value) {
            return detail::get_pointer<T>::apply(value);
        }

        template<typename T> typename scalar_type<T>::type const * get_pointer(T const & value) {
            return detail::get_pointer<T const>::apply(value);
        }

        template<typename T> std::vector<std::size_t> get_extent(T const & value) {
            return detail::get_extent<T>::apply(value);
        }

        template<typename T> void set_extent(T & value, std::vector<std::size_t> const & size) {
            detail::set_extent<T>::apply(value, size);
        }

        template<typename T> bool is_vectorizable(T const & value) {
            return detail::is_vectorizable<T>::apply(value);
        }

        template<typename T> void save(
               archive & ar
             , std::string const & path
             , T const & value
             , std::vector<std::size_t> size = std::vector<std::size_t>()
             , std::vector<std::size_t> chunk = std::vector<std::size_t>()
             , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (chunk.size())
                throw std::logic_error("user defined objects needs to be written continously" + ALPS_STACKTRACE);
            std::string context = ar.get_context();
            ar.set_context(ar.complete_path(path));
            value.save(ar);
            ar.set_context(context);
        }

        template<typename T> void load(
               archive & ar
             , std::string const & path
             , T & value
             , std::vector<std::size_t> chunk = std::vector<std::size_t>()
             , std::vector<std::size_t> offset = std::vector<std::size_t>()
        ) {
            if (chunk.size())
                throw std::logic_error("user defined objects needs to be written continously" + ALPS_STACKTRACE);
            std::string context = ar.get_context();
            ar.set_context(ar.complete_path(path));
            value.load(ar);
            ar.set_context(context);
        }

        #define ALPS_NGS_HDF5_DEFINE_FREE_FUNCTIONS(T)                                                                                                                 \
            template<> struct is_continuous< T >                                                                                                                       \
                : public boost::true_type                                                                                                                              \
            {};                                                                                                                                                        \
            template<> struct is_continuous< T const >                                                                                                                 \
                : public boost::true_type                                                                                                                              \
            {};                                                                                                                                                        \
                                                                                                                                                                       \
            namespace detail {                                                                                                                                         \
                template<> struct ALPS_DECL is_vectorizable< T > {                                                                                                     \
                    static bool apply(T const & value);                                                                                                                \
                };                                                                                                                                                     \
                template<> struct ALPS_DECL is_vectorizable< T const > {                                                                                               \
                    static bool apply(T & value);                                                                                                                      \
                };                                                                                                                                                     \
                                                                                                                                                                       \
                template<> struct ALPS_DECL get_pointer< T > {                                                                                                         \
                    static alps::hdf5::scalar_type< T >::type * apply( T & value);                                                                                     \
                };                                                                                                                                                     \
                                                                                                                                                                       \
                template<> struct ALPS_DECL get_pointer< T const > {                                                                                                   \
                    static alps::hdf5::scalar_type< T >::type const * apply( T const & value);                                                                         \
                };                                                                                                                                                     \
            }                                                                                                                                                          \
                                                                                                                                                                       \
            ALPS_DECL void save(                                                                                                                                       \
                  archive & ar                                                                                                                                         \
                , std::string const & path                                                                                                                             \
                , T const & value                                                                                                                                      \
                , std::vector<std::size_t> size = std::vector<std::size_t>()                                                                                           \
                , std::vector<std::size_t> chunk = std::vector<std::size_t>()                                                                                          \
                , std::vector<std::size_t> offset = std::vector<std::size_t>()                                                                                         \
            );                                                                                                                                                         \
                                                                                                                                                                       \
            ALPS_DECL void load(                                                                                                                                       \
                  archive & ar                                                                                                                                         \
                , std::string const & path                                                                                                                             \
                , T & value                                                                                                                                            \
                , std::vector<std::size_t> chunk = std::vector<std::size_t>()                                                                                          \
                , std::vector<std::size_t> offset = std::vector<std::size_t>()                                                                                         \
            );
        ALPS_NGS_FOREACH_NATIVE_HDF5_TYPE(ALPS_NGS_HDF5_DEFINE_FREE_FUNCTIONS)
        #undef ALPS_NGS_HDF5_DEFINE_FREE_FUNCTIONS

        namespace detail {

            template<typename T> struct make_pvp_proxy {

                explicit make_pvp_proxy(std::string const & path, T value)
                    : path_(path), value_(value)
                {}

                make_pvp_proxy(make_pvp_proxy<T> const & arg)
                    : path_(arg.path_), value_(arg.value_) 
                {}

                std::string path_;
                T value_;
            };

        }

        template <typename T> typename boost::enable_if<
              has_complex_elements<typename alps::detail::remove_cvr<T>::type>
            , archive &
        >::type operator<< (archive & ar, detail::make_pvp_proxy<T> const & proxy) {
            save(ar, proxy.path_, proxy.value_);
            ar.set_complex(proxy.path_);
            return ar;
        }

        template <typename T> typename boost::disable_if<
              has_complex_elements<typename alps::detail::remove_cvr<T>::type>
            , archive &
        >::type operator<< (archive & ar, detail::make_pvp_proxy<T> const & proxy) {
            save(ar, proxy.path_, proxy.value_);
            return ar;
        }

        template <typename T> archive & operator>> (archive & ar, detail::make_pvp_proxy<T> proxy) {
            load(ar, proxy.path_, proxy.value_);
            return ar;
        }
    }

    template <typename T> typename boost::disable_if<typename boost::mpl::and_<
          typename boost::is_same<typename alps::detail::remove_cvr<typename boost::remove_all_extents<T>::type>::type, char>::type
        , typename boost::is_array<T>::type
    >::type, hdf5::detail::make_pvp_proxy<T &> >::type make_pvp(std::string const & path, T & value) {
        return hdf5::detail::make_pvp_proxy<T &>(path, value);
    }
    template <typename T> typename boost::disable_if<typename boost::mpl::and_<
          typename boost::is_same<typename alps::detail::remove_cvr<typename boost::remove_all_extents<T>::type>::type, char>::type
        , typename boost::is_array<T>::type
    >::type, hdf5::detail::make_pvp_proxy<T const &> >::type make_pvp(std::string const & path, T const & value) {
        return hdf5::detail::make_pvp_proxy<T const &>(path, value);
    }
    template <typename T> typename boost::enable_if<typename boost::mpl::and_<
          typename boost::is_same<typename alps::detail::remove_cvr<typename boost::remove_all_extents<T>::type>::type, char>::type
        , typename boost::is_array<T>::type
    >::type, hdf5::detail::make_pvp_proxy<std::string const> >::type make_pvp(std::string const & path, T const & value) {
        return hdf5::detail::make_pvp_proxy<std::string const>(path, value);
    }

    namespace hdf5 {
        namespace detail {

            template<typename A> template<typename T> archive_proxy<A> & archive_proxy<A>::operator=(T const & value) {
                ar_ << make_pvp(path_, value);
                return *this;
            }

            template<typename A> template<typename T> archive_proxy<A> & archive_proxy<A>::operator<<(T const & value) {
                return *this = value;
            }
            
            template<typename A> template <typename T> archive_proxy<A> & archive_proxy<A>::operator>> (T & value) {
                ar_ >> make_pvp(path_, value);
                return *this;
            }

        }
    }
}

#endif
