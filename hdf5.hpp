// Copyright (C) 2008 Lukas Gamper <gamperl -at- gmail.com>
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef ALPS_HDF5
#define ALPS_HDF5

#include <string>
#include <sstream>
#include <cstring>
#include <vector>
#include <complex>
#include <stdexcept>

#include <boost/config.hpp>
#include <boost/utility.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/type_traits.hpp>
#include <boost/function.hpp>

#include <hdf5.h>

namespace alps {
	namespace hdf5 {
		namespace detail {
			class h5_e {
				public:
					static herr_t noop(hid_t) { return 0; }
					static herr_t callback(unsigned n, H5E_error2_t const * desc, void * buffer) {
						*reinterpret_cast<std::ostringstream *>(buffer) << "    #" << n << " " << desc->file_name << " line " << desc->line << " in " << desc->func_name << "(): " << desc->desc << std::endl;
						return 0;
					}
					static void invoke() {
						std::ostringstream buffer;
						buffer << "HDR5 trace:" << std::endl;
						H5Ewalk2(H5E_DEFAULT, H5E_WALK_DOWNWARD, callback, &buffer);
						throw std::runtime_error(buffer.str());
					}
			};
			template<herr_t(*F)(hid_t)> class h5_t {
				public:
					h5_t(): _id(-1) {}
					h5_t(hid_t id): _id(id) {  if (_id < 0) h5_e::invoke(); H5Eclear2(H5E_DEFAULT); }
					~h5_t() { if(_id >= 0 && F(_id) < 0) h5_e::invoke(); H5Eclear2(H5E_DEFAULT); }
					operator hid_t() const { return _id; }
					h5_t & operator=(hid_t id) { if ((_id = id) < 0) h5_e::invoke(); H5Eclear2(H5E_DEFAULT); return *this; }
				private:
					hid_t _id;
			};
			typedef h5_t<H5Fclose> h5f_t;
			typedef h5_t<H5Gclose> h5g_t;
			typedef h5_t<H5Dclose> h5d_t;
			typedef h5_t<H5Aclose> h5a_t;
			typedef h5_t<H5Sclose> h5s_t;
			typedef h5_t<H5Tclose> h5t_t;
			typedef h5_t<H5Pclose> h5p_t;
			typedef h5_t<h5_e::noop> h5e_t;
			template <typename T> class h5_pvp {
				public:
					h5_pvp(std::string const & p, T v): _p(p), _v(v) {}
					h5_pvp(h5_pvp<T> const & c): _p(c._p), _v(c._v) {}
					std::string get_path(std::string const & c) const {
						if (_p.size() && _p[0] == '/')
							return _p;
						else if (_p.size() < 2 || _p.substr(0, 2) != "..")
							return c + (c.size() == 1 ? "" : "/") + _p;
						else {
							std::string s = c;
							std::size_t i = 0;
							for (; s.size() && _p.substr(i, 2) == ".."; i += 3)
								s = s.substr(0, s.find_last_of('/'));
							return s + (s.size() == 1 ? "" : "/") + _p.substr(i);
						}
					}
					T get_value() const { return _v; }
				private:
					std::string _p;
					mutable T _v;
			};
		}
		struct write {};
		struct read {};
		template <typename Tag> class archive: boost::noncopyable {
			public:
				archive(boost::filesystem::path const & file) {
					H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
					hid_t id = H5Fopen(file.native_file_string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
					_file = (id < 0 ? H5Fcreate(file.native_file_string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT) : id);
				}
				~archive() {
					H5Fflush(_file, H5F_SCOPE_GLOBAL);
				}
				void set_context(std::string const & c) {
					_context = c;
				}
				std::string get_context() {
					return _context;
				}
				bool is_group(std::string const & p) const {
					hid_t id = H5Gopen2(_file, p.c_str(), H5P_DEFAULT);
					return id < 0 ? false : static_cast<bool>(detail::h5g_t(id));
				}
				bool is_data(std::string const & p) const {
					hid_t id = H5Dopen2(_file, p.c_str(), H5P_DEFAULT);
					return id < 0 ? false : static_cast<bool>(detail::h5d_t(id));
				}
				std::vector<std::size_t> extent(std::string const & p) const {
					if (is_null(p))
						return std::vector<std::size_t>(1, 0);
					std::vector<hsize_t> buffer(dimensions(p), 0);
					{
						detail::h5d_t data_id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
						detail::h5s_t space_id(H5Dget_space(data_id));
						detail::h5e_t(H5Sget_simple_extent_dims(space_id, &buffer.front(), NULL));
					}
					std::vector<std::size_t> extend(buffer.size(), 0);
					std::copy(buffer.begin(), buffer.end(), extend.begin());
					return extend;
				}
				std::size_t dimensions(std::string const & p) const {
					detail::h5d_t data_id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
					detail::h5s_t space_id(H5Dget_space(data_id));
					return static_cast<hid_t>(detail::h5e_t(H5Sget_simple_extent_dims(space_id, NULL, NULL)));
				}
				bool is_scalar(std::string const & p) const {
					detail::h5d_t data_id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
					detail::h5s_t space_id(H5Dget_space(data_id));
					H5S_class_t type = H5Sget_simple_extent_type(space_id);
					if (type == H5S_NO_CLASS)
						throw std::runtime_error("error reading class " + p);
					return type == H5S_SCALAR;
				}
				bool is_null(std::string const & p) const {
					detail::h5d_t data_id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
					detail::h5s_t space_id(H5Dget_space(data_id));
					H5S_class_t type = H5Sget_simple_extent_type(space_id);
					if (type == H5S_NO_CLASS)
						throw std::runtime_error("error reading class " + p);
					return type == H5S_NULL;
				}
				std::vector<std::string> list_children(std::string const & p) const {
					std::vector<std::string> list;
					H5Giterate2(_file, p.c_str(), NULL, child_visitor, reinterpret_cast<void *>(&list));
					return list;
				}
				std::vector<std::string> list_attr(std::string const & p) const {
					std::vector<std::string> list;
					if (is_group(p)) {
						detail::h5g_t id(H5Gopen2(_file, p.c_str(), H5P_DEFAULT));
						H5Aiterate2(id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, attr_visitor, reinterpret_cast<void *>(&list));
					} else {
						detail::h5d_t id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
						H5Aiterate2(id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, attr_visitor, reinterpret_cast<void *>(&list));
					}
					return list;
				}
				template<typename T> void get_data(std::string const & p, T * v) const {
					detail::h5d_t data_id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
					if (!is_null(p)) {
						detail::h5t_t type_id(get_native_type(v));
						detail::h5e_t(H5Dread2(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, v));
					}
				}
				template<typename T> void get_attr(std::string const & p, std::string const & s, T & v) const {
					if (is_group(p))
						get_attr_helper<detail::h5g_t, T>(H5Gopen2(_file, p.c_str(), H5P_DEFAULT), s, &v);
					else
						get_attr_helper<detail::h5d_t, T>(H5Dopen2(_file, p.c_str(), H5P_DEFAULT), s, &v);
				}
				// = = = = = = = = = = call_serialize = = = = = = = = = =
				template <typename T> void call_serialize(T const & v) {
					v.serialize(*this);
				}
				template <typename T> void call_serialize(T * const & v) {
					v->serialize(*this);
				}
				// = = = = = = = = = = set_data = = = = = = = = = =
				template<typename T> typename boost::enable_if<boost::is_scalar<T> >::type set_data(std::string const & p, T v) {
					detail::h5t_t type_id(get_native_type(v));
					hid_t id = H5Dopen2(_file, p.c_str(), H5P_DEFAULT);
					if (id < 0)
						id = create_path(p, type_id, H5Screate(H5S_SCALAR), 0);
					detail::h5d_t data_id(id);
					detail::h5e_t(H5Dwrite(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v));
				}
				template<typename T> typename boost::disable_if<boost::is_scalar<T> >::type set_data(std::string const & p, T const & v) {
					set_data(p, &const_cast<T &>(v)[0], v.size());
				}
				template<typename T> void set_data(std::string const & p, char const * const & v) {
					set_data(p, std::string(v));
				}
				template<typename T> void set_data(std::string const & p, std::complex<T> const & v) {
					set_data(p, const_cast<T *>(reinterpret_cast<const T *>(&v)), 2);
				}
				template<typename T> void set_data(std::string const & p, T const * v, hsize_t s) {
					detail::h5t_t type_id(get_native_type(v));
					hid_t id = H5Dopen2(_file, p.c_str(), H5P_DEFAULT);
					if (id < 0) {
						id = create_path(p, type_id, s ? H5Screate_simple(1, &s, NULL) : H5Screate(H5S_NULL), s);
					} else 
						detail::h5e_t(H5Dset_extent(id, &s));
					detail::h5d_t data_id(id);
					if (s > 0)
						detail::h5e_t(H5Dwrite(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, v));
				}
				// = = = = = = = = = = append_data = = = = = = = = = =			
				template<typename T> void append_data(std::string const & p, T const * v, hsize_t s) {
					detail::h5t_t type_id(get_native_type(v));
					hid_t id = H5Dopen2(_file, p.c_str(), H5P_DEFAULT);
					if (id < 0)
						return set_data(p, v, s);
					detail::h5d_t data_id(id);
					hsize_t start = extent(p)[0], count = start + s;
					detail::h5e_t(H5Dset_extent(data_id, &count));
					detail::h5s_t space_id(H5Dget_space(data_id));
					detail::h5e_t(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, &start, NULL, &s, NULL));
					detail::h5s_t mem_id(H5Screate_simple(1, &s, NULL));
					detail::h5e_t(H5Dwrite(data_id, type_id, mem_id, space_id, H5P_DEFAULT, v));
				}
				void delete_data(std::string const & p, std::string const & s) {
					detail::h5g_t data_id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
					detail::h5e_t(H5Ldelete(_file, s.c_str(), data_id));
				}
				template<typename T> void set_attr(std::string const & p, std::string const & s, T const & v) {
					if (is_group(p))
						set_attr_helper<detail::h5g_t, T>(H5Gopen2(_file, p.c_str(), H5P_DEFAULT), s, v);
					else
						set_attr_helper<detail::h5d_t, T>(H5Dopen2(_file, p.c_str(), H5P_DEFAULT), s, v);
				}
			private:
				template<typename T> hid_t get_native_type(T &) const { throw std::runtime_error("unknown type"); }
				hid_t get_native_type(char) const { return H5Tcopy(H5T_NATIVE_CHAR); }
				hid_t get_native_type(signed char) const { return H5Tcopy(H5T_NATIVE_SCHAR); }
				hid_t get_native_type(unsigned char) const { return H5Tcopy(H5T_NATIVE_UCHAR); }
				hid_t get_native_type(short) const { return H5Tcopy(H5T_NATIVE_SHORT); }
				hid_t get_native_type(unsigned short) const { return H5Tcopy(H5T_NATIVE_USHORT); }
				hid_t get_native_type(int) const { return H5Tcopy(H5T_NATIVE_INT); }
				hid_t get_native_type(unsigned) const { return H5Tcopy(H5T_NATIVE_UINT); }
				hid_t get_native_type(long) const { return H5Tcopy(H5T_NATIVE_LONG); }
				hid_t get_native_type(unsigned long) const { return H5Tcopy(H5T_NATIVE_ULONG); }
				hid_t get_native_type(long long) const { return H5Tcopy(H5T_NATIVE_LLONG); }
				hid_t get_native_type(unsigned long long) const { return H5Tcopy(H5T_NATIVE_ULLONG); }
				hid_t get_native_type(float) const { return H5Tcopy(H5T_NATIVE_FLOAT); }
				hid_t get_native_type(double) const { return H5Tcopy(H5T_NATIVE_DOUBLE); }
				hid_t get_native_type(long double) const { return H5Tcopy(H5T_NATIVE_LDOUBLE); }
				hid_t get_native_type(bool) const { return H5Tcopy(H5T_NATIVE_HBOOL); }
				template<typename T> hid_t get_native_type(T * ) const { return get_native_type(T()); }
				hid_t get_native_type(char const *) const { return H5Tcopy(H5T_C_S1); }
				static herr_t child_visitor(hid_t, char const * n, void * d) {
					reinterpret_cast<std::vector<std::string> *>(d)->push_back(n);
					return 0;
				}
				static herr_t attr_visitor(hid_t, char const * n, const H5A_info_t *, void * d) {
					reinterpret_cast<std::vector<std::string> *>(d)->push_back(n);
					return 0;
				}
				template<typename I, typename T> void get_attr_helper(I const & data_id, std::string const & s, T * v) const {
					detail::h5t_t type_id(get_native_type(v));
					detail::h5a_t attr_id(H5Aopen2(data_id, s.c_str(), H5P_DEFAULT));
					detail::h5e_t(H5Aread2(attr_id, type_id, v));
				}
				template<typename I, typename T> void set_attr_helper(I const & data_id, std::string const & s, T const & v) {
					detail::h5t_t type_id(get_native_type(v));
					hid_t id = H5Aopen2(data_id, s.c_str(), H5P_DEFAULT);
					if (id < 0) {
						detail::h5s_t space_id(H5Screate(H5S_SCALAR));
						id = H5Acreate2(data_id, s.c_str(), type_id, space_id, H5P_DEFAULT, H5P_DEFAULT);
					}
					detail::h5a_t attr_id(id);
					detail::h5e_t(H5Awrite2(attr_id, type_id, &v));
				}
				hid_t create_path(std::string const & p, hid_t type_id, hid_t space_id, hsize_t s) {
					std::size_t pos;
					hid_t data_id = -1;
					for (pos = p.find_last_of('/'); data_id < 0 && pos > 0 && pos < std::string::npos; pos = p.find_last_of('/', pos - 1))
						data_id = H5Gopen2(_file, p.substr(0, pos).c_str(), H5P_DEFAULT);
					if (data_id < 0) {
						pos = p.find_first_of('/', 1);
						detail::h5g_t(H5Gcreate2(_file, p.substr(0, pos).c_str(), 0, H5P_DEFAULT, H5P_DEFAULT));
					} else {
						pos = p.find_first_of('/', pos + 1);
						detail::h5g_t(data_id);
					}
					while ((pos = p.find_first_of('/', pos + 1)) != std::string::npos && pos > 0)
						detail::h5g_t(H5Gcreate2(_file, p.substr(0, pos).c_str(), 0, H5P_DEFAULT, H5P_DEFAULT));
					detail::h5p_t prop_id(H5Pcreate(H5P_DATASET_CREATE)); 
					detail::h5e_t(H5Pset_fill_time(prop_id, H5D_FILL_TIME_NEVER));
					if (s > 0)
						detail::h5e_t(H5Pset_chunk (prop_id, 1, &s));
					return H5Dcreate2(_file, p.c_str(), type_id, detail::h5s_t(space_id), H5P_DEFAULT, prop_id, H5P_DEFAULT);
				}
				std::string _context;
				detail::h5f_t _file;
		};
		namespace detail {
			template <typename T, void (T::* F)(archive<write> &) const> struct h5_has_serialize_wrapper {
				typedef T type; 
			};
			template <typename T> boost::type_traits::yes_type h5_has_serialize_helper(
				typename h5_has_serialize_wrapper<T, &T::serialize>::type *
			);
			template <typename T> boost::type_traits::no_type h5_has_serialize_helper(...);
			template<typename T, bool B> struct h5_has_serialize {
				enum { value = false };
			};
			template<typename T> struct h5_has_serialize<T, true> {
				enum { value = sizeof(h5_has_serialize_helper<T>(NULL)) == sizeof(boost::type_traits::yes_type) };
			};
			template<typename T> struct h5_has_serialize <T *, true>{
				enum { value = sizeof(h5_has_serialize_helper<T>(NULL)) == sizeof(boost::type_traits::yes_type) };
			};
			template<typename T> struct is_class : public boost::is_class<T> {};
			template<typename T> struct is_class<T *> : public boost::is_class<T> {};
		}
		template <typename T> typename boost::enable_if<
			boost::mpl::bool_<detail::h5_has_serialize<T, detail::is_class<T>::value>::value>, archive<write> &
		>::type operator<< (archive<write> & ar, detail::h5_pvp<T> const & v) {
			std::string c = ar.get_context();
			ar.set_context(v.get_path(ar.get_context()));
			ar.call_serialize(v.get_value());
			ar.set_context(c);
			return ar;
		}
		template <typename T> typename boost::disable_if<
			boost::mpl::bool_<detail::h5_has_serialize<T, detail::is_class<T>::value>::value>, archive<write> &
		>::type operator<< (archive<write> & ar, detail::h5_pvp<T> const & v) {
			ar.set_data(v.get_path(ar.get_context()), v.get_value());
			return ar;
		}
		template <typename T> archive<read> & operator>> (archive<read> & ar, detail::h5_pvp<T &> const & v) {
//			ar.get_data(v.get_path(ar.get_context()), v.get_value());
//			return ar;
		}
	}
	template <typename T> hdf5::detail::h5_pvp<T &> make_pvp(std::string const & p, T & v) {
		return hdf5::detail::h5_pvp<T &>(p, v);
	}
	template <typename T> hdf5::detail::h5_pvp<T> make_pvp(std::string const & p, T const & v) {
		return hdf5::detail::h5_pvp<T>(p, v);
	}
}
#endif
