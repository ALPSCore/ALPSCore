/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include <hdf5.h>

#ifdef ALPS_SINGLE_THREAD
    #define ALPS_HDF5_LOCK_MUTEX
#else
    #define ALPS_HDF5_LOCK_MUTEX boost::lock_guard<boost::recursive_mutex> guard(mutex_);
#endif

#ifdef H5_HAVE_THREADSAFE
    #define ALPS_HDF5_FAKE_THREADSAFETY
#else
    #define ALPS_HDF5_FAKE_THREADSAFETY ALPS_HDF5_LOCK_MUTEX
#endif

#define ALPS_HDF5_NATIVE_INTEGRAL_TYPES   \
    char, signed char, unsigned char,     \
    short, unsigned short,                \
    int, unsigned, long, unsigned long,   \
    long long, unsigned long long,        \
    float, double, long double,           \
    bool

#define ALPS_HDF5_FOREACH_NATIVE_TYPE_INTEGRAL(CALLBACK, ARG)                                                                                                           \
    CALLBACK(char, ARG)                                                                                                                                                 \
    CALLBACK(signed char, ARG)                                                                                                                                          \
    CALLBACK(unsigned char, ARG)                                                                                                                                        \
    CALLBACK(short, ARG)                                                                                                                                                \
    CALLBACK(unsigned short, ARG)                                                                                                                                       \
    CALLBACK(int, ARG)                                                                                                                                                  \
    CALLBACK(unsigned, ARG)                                                                                                                                             \
    CALLBACK(long, ARG)                                                                                                                                                 \
    CALLBACK(unsigned long, ARG)                                                                                                                                        \
    CALLBACK(long long, ARG)                                                                                                                                            \
    CALLBACK(unsigned long long, ARG)                                                                                                                                   \
    CALLBACK(float, ARG)                                                                                                                                                \
    CALLBACK(double, ARG)                                                                                                                                               \
    CALLBACK(long double, ARG)                                                                                                                                          \
    CALLBACK(bool, ARG)

namespace alps {
    namespace hdf5 {
        namespace detail {

            template<typename T> struct native_ptr_converter {
                native_ptr_converter(std::size_t) {}
                inline T const * apply(T const * v) {
                    return v;
                }
            };

            template<> struct native_ptr_converter<std::string> {
                std::vector<char const *> data;
                native_ptr_converter(std::size_t size): data(size) {}
                inline char const * const * apply(std::string const * v) {
                    for (std::vector<char const *>::iterator it = data.begin(); it != data.end(); ++it)
                            *it = v[it - data.begin()].c_str();
                    return &data[0];
                }
            };

            inline herr_t noop(hid_t) {
                return 0;
            }

            class error {

                public:

                    std::string invoke(hid_t id) {
                        std::ostringstream buffer;
                        buffer << "HDF5 error: " << cast<std::string>(id) << std::endl;
                        H5Ewalk2(H5E_DEFAULT, H5E_WALK_DOWNWARD, callback, &buffer);
                        return buffer.str();
                    }

                private:

                    static herr_t callback(unsigned n, H5E_error2_t const * desc, void * buffer) {
                        *reinterpret_cast<std::ostringstream *>(buffer)
                            << "    #"
                            << cast<std::string>(n)
                            << " " << desc->file_name
                            << " line "
                            << cast<std::string>(desc->line)
                            << " in "
                            << desc->func_name
                            << "(): "
                            << desc->desc
                            << std::endl;
                        return 0;
                    }

            };

            template<herr_t(*F)(hid_t)> class resource {
                public:
                    resource(): _id(-1) {}
                    resource(hid_t id): _id(id) {
                        if (_id < 0)
                            throw archive_error(error().invoke(_id) + ALPS_STACKTRACE);
                    }

                    ~resource() {
                        if(_id < 0 || (_id = F(_id)) < 0) {
                            std::cerr << "Error in "
                                      << __FILE__
                                      << " on "
                                      << ALPS_STRINGIFY(__LINE__)
                                      << " in "
                                      << __FUNCTION__ // TODO: check for gcc and use __PRETTY_FUNCTION__
                                      << ":"
                                      << std::endl
                                      << error().invoke(_id)
                                      << std::endl;
                            std::abort();
                        }
                    }

                    operator hid_t() const {
                        return _id;
                    }

                    resource<F> & operator=(hid_t id) {
                        if ((_id = id) < 0)
                            throw archive_error(error().invoke(_id) + ALPS_STACKTRACE);
                        return *this;
                    }

                private:
                    hid_t _id;
            };

            typedef resource<H5Gclose> group_type;
            typedef resource<H5Dclose> data_type;
            typedef resource<H5Aclose> attribute_type;
            typedef resource<H5Sclose> space_type;
            typedef resource<H5Tclose> type_type;
            typedef resource<H5Pclose> property_type;
            typedef resource<noop> error_type;

            inline hid_t check_group(hid_t id) { group_type unused(id); return unused; }
            inline hid_t check_data(hid_t id) { data_type unused(id); return unused; }
            inline hid_t check_attribute(hid_t id) { attribute_type unused(id); return unused; }
            inline hid_t check_space(hid_t id) { space_type unused(id); return unused; }
            inline hid_t check_type(hid_t id) { type_type unused(id); return unused; }
            inline hid_t check_property(hid_t id) { property_type unused(id); return unused; }
            inline hid_t check_error(hid_t id) { error_type unused(id); return unused; }

            inline hid_t get_native_type(char) { return H5Tcopy(H5T_NATIVE_CHAR); }
            inline hid_t get_native_type(signed char) { return H5Tcopy(H5T_NATIVE_SCHAR); }
            inline hid_t get_native_type(unsigned char) { return H5Tcopy(H5T_NATIVE_UCHAR); }
            inline hid_t get_native_type(short) { return H5Tcopy(H5T_NATIVE_SHORT); }
            inline hid_t get_native_type(unsigned short) { return H5Tcopy(H5T_NATIVE_USHORT); }
            inline hid_t get_native_type(int) { return H5Tcopy(H5T_NATIVE_INT); }
            inline hid_t get_native_type(unsigned) { return H5Tcopy(H5T_NATIVE_UINT); }
            inline hid_t get_native_type(long) { return H5Tcopy(H5T_NATIVE_LONG); }
            inline hid_t get_native_type(unsigned long) { return H5Tcopy(H5T_NATIVE_ULONG); }
            inline hid_t get_native_type(long long) { return H5Tcopy(H5T_NATIVE_LLONG); }
            inline hid_t get_native_type(unsigned long long) { return H5Tcopy(H5T_NATIVE_ULLONG); }
            inline hid_t get_native_type(float) { return H5Tcopy(H5T_NATIVE_FLOAT); }
            inline hid_t get_native_type(double) { return H5Tcopy(H5T_NATIVE_DOUBLE); }
            inline hid_t get_native_type(long double) { return H5Tcopy(H5T_NATIVE_LDOUBLE); }
            inline hid_t get_native_type(bool) { return H5Tcopy(H5T_NATIVE_SCHAR); }
            inline hid_t get_native_type(std::string) {
                hid_t type_id = H5Tcopy(H5T_C_S1);
                detail::check_error(H5Tset_size(type_id, H5T_VARIABLE));
                return type_id;
            }
        }
    }
}
