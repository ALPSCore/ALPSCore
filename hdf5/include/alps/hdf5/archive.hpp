/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_HDF5_ARCHIVE_HPP
#define ALPS_HDF5_ARCHIVE_HPP

#include <alps/hdf5/config.hpp>
#include <alps/utilities/stacktrace.hpp>
#include <alps/hdf5/errors.hpp>
#include <alps/utilities/remove_cvr.hpp>
#include <alps/utilities/type_wrapper.hpp>

// FIXME: remove together with deprecated methods
#include <alps/utilities/deprecated.hpp>

#ifndef ALPS_SINGLE_THREAD

#include <boost/thread.hpp>

#endif

#include <map>
#include <vector>
#include <string>
#include <type_traits>
#include <numeric>

#define ALPS_FOREACH_NATIVE_HDF5_TYPE(CALLBACK)                                                                                                                        \
    CALLBACK(char)                                                                                                                                                     \
    CALLBACK(signed char)                                                                                                                                              \
    CALLBACK(unsigned char)                                                                                                                                            \
    CALLBACK(short)                                                                                                                                                    \
    CALLBACK(unsigned short)                                                                                                                                           \
    CALLBACK(int)                                                                                                                                                      \
    CALLBACK(unsigned)                                                                                                                                                 \
    CALLBACK(long)                                                                                                                                                     \
    CALLBACK(unsigned long)                                                                                                                                            \
    CALLBACK(long long)                                                                                                                                                \
    CALLBACK(unsigned long long)                                                                                                                                       \
    CALLBACK(float)                                                                                                                                                    \
    CALLBACK(double)                                                                                                                                                   \
    CALLBACK(long double)                                                                                                                                              \
    CALLBACK(bool)                                                                                                                                                     \
    CALLBACK(std::string)

namespace alps {
    namespace hdf5 {

        /// Inherits from `true_type` if `T` is a native type, from `false_type` otherwise
        template <typename T>
        struct is_native_type : public std::false_type {};
#define ALPS_HDF5_IS_NATIVE_TYPE_CALLER(__type__)    \
        template <> struct is_native_type<__type__> : public std::true_type {};
        ALPS_FOREACH_NATIVE_HDF5_TYPE(ALPS_HDF5_IS_NATIVE_TYPE_CALLER)
#undef ALPS_HDF5_IS_NATIVE_TYPE_CALLER

#define ONLY_NATIVE(T,R) typename std::enable_if<is_native_type<T>::value, R>::type
#define ONLY_NOT_NATIVE(T,R) typename std::enable_if<!is_native_type<T>::value, R>::type

        namespace detail {
            struct archivecontext;

            template<typename A, typename T> struct is_datatype_caller {
                static bool apply(A const & ar, std::string path) {
                    throw std::logic_error("only native datatypes can be probed: " + path + ALPS_STACKTRACE);
                    return false;
                }
            };

            #define ALPS_HDF5_IS_DATATYPE_CALLER(T)                                                                                                                    \
                template<typename A> struct is_datatype_caller<A, T > {                                                                                                \
                    static bool apply(A const & ar, std::string path, T unused = alps::detail::type_wrapper<T>::type()) {                                              \
                        return ar.is_datatype_impl(path, unused);                                                                                                      \
                    }                                                                                                                                                  \
                };
            ALPS_FOREACH_NATIVE_HDF5_TYPE(ALPS_HDF5_IS_DATATYPE_CALLER)
            #undef ALPS_HDF5_IS_DATATYPE_CALLER

            template<typename A> struct archive_proxy {

                explicit archive_proxy(std::string const & path, A & ar)
                    : path_(path), ar_(ar)
                {}

                template<typename T> archive_proxy & operator=(T const & value);
                template<typename T> archive_proxy & operator<<(T const & value);
                template<typename T> archive_proxy & operator>>(T & value);

                std::string path_;
                A ar_;
            };
        }

        class archive {
            private:
               /// Assignment operator is deleted
               archive& operator=(const archive&); /* not implemented*/ // FIXME:TODO:C++11 `=delete` or implement via `swap()`
            // FIXME: MAKE private:
            public:
                typedef enum {
                    READ = 0x00,
                    WRITE = 0x01,
                    REPLACE = 0x02,
                    COMPRESS = 0x04,
                    MEMORY = 0x10
                } properties;

            public:

                /// default constructor to create archive with out openning of any file
                /// to be used in conjunction with `void open(const std::string &, std::string)` function
                archive();
                archive(std::string const & filename, std::string mode = "r");
                archive(std::string const & filename, int prop)  ALPS_DEPRECATED;
                archive(archive const & arg);

                virtual ~archive();
                static void abort();

                std::string const & get_filename() const;

                std::string encode_segment(std::string segment) const;
                std::string decode_segment(std::string segment) const;

                std::string get_context() const;
                void set_context(std::string const & context);
                std::string complete_path(std::string path) const;
                /// open a new archive file
                /// check that the archive is not already opened and construct archive.
                void open(const std::string & filename, const std::string &mode = "r");
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

                template<typename T> auto read(
                      std::string path
                    , T *
                    , std::vector<std::size_t>
                    , std::vector<std::size_t> = std::vector<std::size_t>()
                ) const -> ONLY_NOT_NATIVE(T, void) {
                    throw std::logic_error("Invalid type on path: " + path + ALPS_STACKTRACE);
                }

                template<typename T> auto write(
                      std::string path
                    , T const * value
                    , std::vector<std::size_t> size
                    , std::vector<std::size_t> chunk = std::vector<std::size_t>()
                    , std::vector<std::size_t> offset = std::vector<std::size_t>()
                ) const -> ONLY_NOT_NATIVE(T, void) {
                    throw std::logic_error("Invalid type on path: " + path + ALPS_STACKTRACE);
                }

                template<typename T> auto read(std::string path, T & value) const -> ONLY_NATIVE(T, void);

                template<typename T> auto read(std::string path
                                             , T * value
                                             , std::vector<std::size_t> chunk
                                             , std::vector<std::size_t> offset = std::vector<std::size_t>()
                    ) const -> ONLY_NATIVE(T, void);

                template<typename T> auto write(std::string path, T value) const -> ONLY_NATIVE(T, void);

                template<typename T> auto write(std::string path
                                              , T const * value, std::vector<std::size_t> size
                                              , std::vector<std::size_t> chunk = std::vector<std::size_t>()
                                              , std::vector<std::size_t> offset = std::vector<std::size_t>()
                    ) const -> ONLY_NATIVE(T, void);

                template<typename T> auto is_datatype_impl(std::string path, T) const -> ONLY_NATIVE(T, bool);

            private:

                void construct(std::string const & filename, std::size_t props = READ);
                std::string file_key(std::string filename, bool memory) const;

                std::string current_;
                detail::archivecontext * context_;

#ifndef ALPS_SINGLE_THREAD
                static boost::recursive_mutex mutex_;
#endif
                static std::map<std::string, std::pair<detail::archivecontext *, std::size_t> > ref_cnt_;

        };

        template<typename T> struct is_continuous
            : public std::false_type
        {};

        template<typename T> struct is_content_continuous
            : public is_continuous<T>
        {};

        template<typename T> struct has_complex_elements
            : public std::false_type
        {};

        template<typename T> struct scalar_type {
            typedef T type;
        };

        namespace detail {

             template<typename T> struct get_extent {
                static std::vector<std::size_t> apply(T const & /*value*/) {
                    return std::vector<std::size_t>();
                }
            };

            template<typename T> struct set_extent {
                 static void apply(T &, std::vector<std::size_t> const &) {}
            };

            #define ALPS_HDF5_DEFINE_SET_EXTENT(T)                                                                                                                  \
                template<> struct set_extent<T> {                                                                                                                   \
                    static void apply(T &, std::vector<std::size_t> const & extent) {                                                                               \
                        if (extent.size() > 0)                                                                                                                      \
                            throw wrong_type("The extents do not match" + ALPS_STACKTRACE);                                                                         \
                    }                                                                                                                                               \
                };
            ALPS_FOREACH_NATIVE_HDF5_TYPE(ALPS_HDF5_DEFINE_SET_EXTENT)
            #undef ALPS_HDF5_DEFINE_SET_EXTENT

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
             , std::vector<std::size_t> /*size*/ = std::vector<std::size_t>()
             , std::vector<std::size_t> chunk = std::vector<std::size_t>()
             , std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()
        ) {
            // FIXME: size and offset are unused -- we should at least check for this
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
             , std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()
        ) {
            if (chunk.size())
                throw std::logic_error("user defined objects needs to be written continously" + ALPS_STACKTRACE);
            std::string context = ar.get_context();
            ar.set_context(ar.complete_path(path));
            value.load(ar);
            ar.set_context(context);
        }

        #define ALPS_HDF5_DEFINE_FREE_FUNCTIONS(T)                                                                                                                     \
            template<> struct is_continuous< T >                                                                                                                       \
                : public std::true_type                                                                                                                                \
            {};                                                                                                                                                        \
            template<> struct is_continuous< T const >                                                                                                                 \
                : public std::true_type                                                                                                                                \
            {};                                                                                                                                                        \
                                                                                                                                                                       \
            namespace detail {                                                                                                                                         \
                template<> struct is_vectorizable< T > {                                                                                                               \
                    static bool apply(T const & value);                                                                                                                \
                };                                                                                                                                                     \
                template<> struct is_vectorizable< T const > {                                                                                                         \
                    static bool apply(T & value);                                                                                                                      \
                };                                                                                                                                                     \
                                                                                                                                                                       \
                template<> struct get_pointer< T > {                                                                                                                   \
                    static alps::hdf5::scalar_type< T >::type * apply( T & value);                                                                                     \
                };                                                                                                                                                     \
                                                                                                                                                                       \
                template<> struct get_pointer< T const > {                                                                                                             \
                    static alps::hdf5::scalar_type< T >::type const * apply( T const & value);                                                                         \
                };                                                                                                                                                     \
            }                                                                                                                                                          \
                                                                                                                                                                       \
            void save(                                                                                                                                                 \
                  archive & ar                                                                                                                                         \
                , std::string const & path                                                                                                                             \
                , T const & value                                                                                                                                      \
                , std::vector<std::size_t> size = std::vector<std::size_t>()                                                                                           \
                , std::vector<std::size_t> chunk = std::vector<std::size_t>()                                                                                          \
                , std::vector<std::size_t> offset = std::vector<std::size_t>()                                                                                         \
            );                                                                                                                                                         \
                                                                                                                                                                       \
            void load(                                                                                                                                                 \
                  archive & ar                                                                                                                                         \
                , std::string const & path                                                                                                                             \
                , T & value                                                                                                                                            \
                , std::vector<std::size_t> chunk = std::vector<std::size_t>()                                                                                          \
                , std::vector<std::size_t> offset = std::vector<std::size_t>()                                                                                         \
            );
        ALPS_FOREACH_NATIVE_HDF5_TYPE(ALPS_HDF5_DEFINE_FREE_FUNCTIONS)
        #undef ALPS_HDF5_DEFINE_FREE_FUNCTIONS

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

        template <typename T> typename std::enable_if<
              has_complex_elements<typename alps::detail::remove_cvr<T>::type>::value
            , archive &
        >::type operator<< (archive & ar, detail::make_pvp_proxy<T> const & proxy) {
            save(ar, proxy.path_, proxy.value_);
            ar.set_complex(proxy.path_);
            return ar;
        }

        template <typename T> typename std::enable_if<
              !has_complex_elements<typename alps::detail::remove_cvr<T>::type>::value
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

    template <typename T> typename std::enable_if<!(
          std::is_same<typename alps::detail::remove_cvr<typename std::remove_all_extents<T>::type>::type, char>::value
       && std::is_array<T>::value)
       , hdf5::detail::make_pvp_proxy<T &> >::type make_pvp(std::string const & path, T & value) {
        return hdf5::detail::make_pvp_proxy<T &>(path, value);
    }
    template <typename T> typename std::enable_if<!(
          std::is_same<typename alps::detail::remove_cvr<typename std::remove_all_extents<T>::type>::type, char>::value
       && std::is_array<T>::value)
       , hdf5::detail::make_pvp_proxy<T const &> >::type make_pvp(std::string const & path, T const & value) {
        return hdf5::detail::make_pvp_proxy<T const &>(path, value);
    }
    template <typename T> typename std::enable_if<
          std::is_same<typename alps::detail::remove_cvr<typename std::remove_all_extents<T>::type>::type, char>::value
       && std::is_array<T>::value
       , hdf5::detail::make_pvp_proxy<std::string const> >::type make_pvp(std::string const & path, T const & value) {
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
