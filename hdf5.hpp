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

#ifdef ALPS_HAVE_VALARRAY
# include <valarray>
#endif

#include <boost/config.hpp>
#include <boost/utility.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/type_traits.hpp>
#include <boost/function.hpp>

#include <hdf5.h>

namespace alps {
	namespace hdf5 {
		namespace detail {
			class error {
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
			template<herr_t(*F)(hid_t)> class ressource {
				public:
					ressource(): _id(-1) {}
					ressource(hid_t id): _id(id) {  if (_id < 0) error::invoke(); H5Eclear(H5E_DEFAULT); }
					~ressource() { if(_id >= 0 && F(_id) < 0) error::invoke(); H5Eclear(H5E_DEFAULT); }
					operator hid_t() const { return _id; }
					ressource & operator=(hid_t id) { if ((_id = id) < 0) error::invoke(); H5Eclear2(H5E_DEFAULT); return *this; }
				private:
					hid_t _id;
			};
			typedef ressource<H5Fclose> file_type;
			typedef ressource<H5Gclose> group_type;
			typedef ressource<H5Dclose> data_type;
			typedef ressource<H5Aclose> attribute_type;
			typedef ressource<H5Sclose> space_type;
			typedef ressource<H5Tclose> type_type;
			typedef ressource<H5Pclose> property_type;
			typedef ressource<error::noop> error_type;
		}
		struct write {};
		struct read {};
		struct serializable {};
		template <typename Tag> class archive: boost::noncopyable {
			public:
				archive() {
					throw std::runtime_error("no filename passed");
				}
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
				std::string compute_path(std::string const & p) {
					if (p.size() && p[0] == '/')
						return p;
					else if (p.size() < 2 || p.substr(0, 2) != "..")
						return _context + (_context.size() == 1 ? "" : "/") + p;
					else {
						std::string s = _context;
						std::size_t i = 0;
						for (; s.size() && p.substr(i, 2) == ".."; i += 3)
							s = s.substr(0, s.find_last_of('/'));
						return s + (s.size() == 1 ? "" : "/") + p.substr(i);
					}
				}
				bool is_group(std::string const & p) const {
					hid_t id = H5Gopen2(_file, p.c_str(), H5P_DEFAULT);
					return id < 0 ? false : static_cast<bool>(detail::group_type(id));
				}
				bool is_data(std::string const & p) const {
					hid_t id = H5Dopen2(_file, p.c_str(), H5P_DEFAULT);
					return id < 0 ? false : static_cast<bool>(detail::data_type(id));
				}
				std::vector<std::size_t> extent(std::string const & p) const {
					if (is_null(p))
						return std::vector<std::size_t>(1, 0);
					std::vector<hsize_t> buffer(dimensions(p), 0);
					{
						detail::data_type data_id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
						detail::space_type space_id(H5Dget_space(data_id));
						detail::error_type(H5Sget_simple_extent_dims(space_id, &buffer.front(), NULL));
					}
					std::vector<std::size_t> extend(buffer.size(), 0);
					std::copy(buffer.begin(), buffer.end(), extend.begin());
					return extend;
				}
				std::size_t dimensions(std::string const & p) const {
					detail::data_type data_id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
					detail::space_type space_id(H5Dget_space(data_id));
					return static_cast<hid_t>(detail::error_type(H5Sget_simple_extent_dims(space_id, NULL, NULL)));
				}
				bool is_scalar(std::string const & p) const {
					detail::data_type data_id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
					detail::space_type space_id(H5Dget_space(data_id));
					H5S_class_t type = H5Sget_simple_extent_type(space_id);
					if (type == H5S_NO_CLASS)
						throw std::runtime_error("error reading class " + p);
					return type == H5S_SCALAR;
				}
				bool is_null(std::string const & p) const {
					detail::data_type data_id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
					detail::space_type space_id(H5Dget_space(data_id));
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
						detail::group_type id(H5Gopen2(_file, p.c_str(), H5P_DEFAULT));
						H5Aiterate2(id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, attr_visitor, reinterpret_cast<void *>(&list));
					} else {
						detail::data_type id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
						H5Aiterate2(id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, attr_visitor, reinterpret_cast<void *>(&list));
					}
					return list;
				}
// = = = = = = = = = = get_data = = = = = = = = = =
				template<typename T> void get_data(std::string const & p, T * v) const {
					detail::data_type data_id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
					if (!is_null(p)) {
						detail::type_type type_id(get_native_type(v));
						detail::error_type(H5Dread2(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, v));
					}
				}
// = = = = = = = = = = get_attr = = = = = = = = = =
				template<typename T> void get_attr(std::string const & p, std::string const & s, T & v) const {
					if (is_group(p))
						get_attr_helper<detail::group_type, T>(H5Gopen2(_file, p.c_str(), H5P_DEFAULT), s, &v);
					else
						get_attr_helper<detail::data_type, T>(H5Dopen2(_file, p.c_str(), H5P_DEFAULT), s, &v);
				}
// = = = = = = = = = = set_data = = = = = = = = = =
				template<typename T> typename boost::enable_if<boost::is_scalar<T> >::type set_data(std::string const & p, T v) {
					detail::type_type type_id(get_native_type(v));
					hid_t id = H5Dopen2(_file, p.c_str(), H5P_DEFAULT);
					if (id < 0)
						id = create_path(p, type_id, H5Screate(H5S_SCALAR), 0);
					detail::data_type data_id(id);
					detail::error_type(H5Dwrite(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v));
				}
				template<typename T> typename boost::disable_if<boost::is_scalar<T> >::type set_data(std::string const & p, T const & v) {
					set_data(p, &const_cast<T &>(v)[0], v.size());
				}
				template<typename T> void set_data(std::string const & p, char const * const & v) {
					set_data(p, std::string(v));
				}
				template<typename T> void set_data(std::string const & p, std::complex<T> const & v) {
					set_data(p, &v, 1);
				}
				#ifdef ALPS_HAVE_VALARRAY
					template<typename T> void set_data(std::string const & p, std::vector<std::valarray<T> > const & v) {
					
					
					std::cout << "Not impl: " << p << std::endl;
					
	//					set_data(p, const_cast<T *>(reinterpret_cast<const T *>(&v)), 2);
	
	
	
					}
				#endif
				template<typename T> void set_data(std::string const & p, T const * v, hsize_t s) {
					detail::type_type type_id(get_native_type(v));
					hid_t id = H5Dopen2(_file, p.c_str(), H5P_DEFAULT);
					if (id < 0) {
						id = create_path(p, type_id, s ? H5Screate_simple(1, &s, NULL) : H5Screate(H5S_NULL), s);
					} else 
						detail::error_type(H5Dset_extent(id, &s));
					detail::data_type data_id(id);
					if (s > 0)
						detail::error_type(H5Dwrite(data_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, v));
				}
				template<typename T> void set_data(std::string const & p, std::complex<T> const * v, hsize_t s) {
					set_data(p, static_cast<T const *>(v), 2 * s);
				}
// = = = = = = = = = = append_data = = = = = = = = = =			
				template<typename T> void append_data(std::string const & p, T const * v, hsize_t s) {
					detail::type_type type_id(get_native_type(v));
					hid_t id = H5Dopen2(_file, p.c_str(), H5P_DEFAULT);
					if (id < 0)
						return set_data(p, v, s);
					detail::data_type data_id(id);
					hsize_t start = extent(p)[0], count = start + s;
					detail::error_type(H5Dset_extent(data_id, &count));
					detail::space_type space_id(H5Dget_space(data_id));
					detail::error_type(H5Sselect_hyperslab(space_id, H5S_SELECT_SET, &start, NULL, &s, NULL));
					detail::space_type mem_id(H5Screate_simple(1, &s, NULL));
					detail::error_type(H5Dwrite(data_id, type_id, mem_id, space_id, H5P_DEFAULT, v));
				}
// = = = = = = = = = = delete_data = = = = = = = = = =			
				void delete_data(std::string const & p, std::string const & s) {
					detail::group_type data_id(H5Dopen2(_file, p.c_str(), H5P_DEFAULT));
					detail::error_type(H5Ldelete(_file, s.c_str(), data_id));
				}
// = = = = = = = = = = set_attr = = = = = = = = = =			
				template<typename T> void set_attr(std::string const & p, std::string const & s, T const & v) {
					if (is_group(p))
						set_attr_helper<detail::group_type, T>(H5Gopen2(_file, p.c_str(), H5P_DEFAULT), s, v);
					else
						set_attr_helper<detail::data_type, T>(H5Dopen2(_file, p.c_str(), H5P_DEFAULT), s, v);
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
					detail::type_type type_id(get_native_type(v));
					detail::attribute_type attr_id(H5Aopen(data_id, s.c_str(), H5P_DEFAULT));
					detail::error_type(H5Aread(attr_id, type_id, v));
				}
				template<typename I, typename T> void set_attr_helper(I const & data_id, std::string const & s, T const & v) {
					detail::type_type type_id(get_native_type(v));
					hid_t id = H5Aopen(data_id, s.c_str(), H5P_DEFAULT);
					if (id < 0) {
						detail::space_type space_id(H5Screate(H5S_SCALAR));
						id = H5Acreate(data_id, s.c_str(), type_id, space_id, H5P_DEFAULT, H5P_DEFAULT);
					}
					detail::attribute_type attr_id(id);
					detail::error_type(H5Awrite(attr_id, type_id, &v));
				}
				hid_t create_path(std::string const & p, hid_t type_id, hid_t space_id, hsize_t s) {
					std::size_t pos;
					hid_t data_id = -1;
					for (pos = p.find_last_of('/'); data_id < 0 && pos > 0 && pos < std::string::npos; pos = p.find_last_of('/', pos - 1))
						data_id = H5Gopen2(_file, p.substr(0, pos).c_str(), H5P_DEFAULT);
					if (data_id < 0) {
						pos = p.find_first_of('/', 1);
						detail::group_type(H5Gcreate2(_file, p.substr(0, pos).c_str(), 0, H5P_DEFAULT, H5P_DEFAULT));
					} else {
						pos = p.find_first_of('/', pos + 1);
						detail::group_type(data_id);
					}
					while ((pos = p.find_first_of('/', pos + 1)) != std::string::npos && pos > 0)
						detail::group_type(H5Gcreate2(_file, p.substr(0, pos).c_str(), 0, H5P_DEFAULT, H5P_DEFAULT));
					detail::property_type prop_id(H5Pcreate(H5P_DATASET_CREATE)); 
					detail::error_type(H5Pset_fill_time(prop_id, H5D_FILL_TIME_NEVER));
					if (s > 0)
						detail::error_type(H5Pset_chunk (prop_id, 1, &s));
					return H5Dcreate2(_file, p.c_str(), type_id, detail::space_type(space_id), H5P_DEFAULT, prop_id, H5P_DEFAULT);
				}
				std::string _context;
				detail::file_type _file;
		};
		namespace detail {
			typedef enum {plain, ptr, array} pvp_type;
			template <typename T, pvp_type U = plain> class pvp {
				public:
					pvp(std::string const & p, T v): _p(p), _v(v) {}
					pvp(pvp<T, U> const & c): _p(c._p), _v(c._v) {}
					std::string get_path() const { return _p; }
					void serialize(hdf5::archive<hdf5::write> & ar) const { _v.serialize(ar); }
					void set_value(hdf5::archive<hdf5::write> & ar) const { 
						if (_p.find_last_of('@') != std::string::npos) {
							std::string s = _p.substr(_p.find_last_of('@') + 1);
							std::string p = _p.substr(0, _p.find_last_of('@') - 1);
							ar.set_attr(p, s, _v);
						} else
							ar.set_data(ar.compute_path(_p), _v);
					}
				private:
					std::string _p;
					mutable T _v;
			};
			template <typename T> class pvp<T, ptr> {
				public:
					pvp(std::string const & p, T * v): _p(p), _v(v) {}
					pvp(pvp<T, ptr> const & c): _p(c._p), _v(c._v) {}
					std::string get_path() const { return _p; }
					void serialize(hdf5::archive<hdf5::write> & ar) const { _v->serialize(ar); }
					void set_value(hdf5::archive<hdf5::write> & ar) const { ar.set_data(ar.compute_path(_p), *_v); }
				private:
					std::string _p;
					mutable T * _v;
			};
			template <typename T> class pvp<T, array> {
				public:
					pvp(std::string const & p, T * v, std::size_t s): _p(p), _v(v), _s(s) {}
					pvp(pvp<T, ptr> const & c): _p(c._p), _v(c._v), _s(c._s) {}
					std::string get_path() const { return _p; }
					void serialize(hdf5::archive<hdf5::write> & ar) const { _v->serialize(ar); }
					void set_value(hdf5::archive<hdf5::write> & ar) const { ar.set_data(ar.compute_path(_p), _v, _s); }
				private:
					std::string _p;
					std::size_t _s;
					mutable T * _v;
			};
			template <typename T, void (T::*)(archive<write> &) const> struct has_serialize_wrapper {
				typedef T type;
			};
			template <typename T> boost::type_traits::yes_type has_serialize_helper(
				typename has_serialize_wrapper<T, &T::serialize>::type *
			);
			template <typename T> boost::type_traits::no_type has_serialize_helper(...);
			template<typename T, bool B> struct has_serialize {
				enum { value = false };
			};
			template<typename T> struct has_serialize<T, true> {
				enum { value = sizeof(has_serialize_helper<T>(NULL)) == sizeof(boost::type_traits::yes_type) };
			};
		}
		template <typename T, detail::pvp_type U> typename boost::enable_if<boost::mpl::bool_<
			boost::is_base_of<serializable, typename boost::remove_reference<typename boost::remove_cv<T>::type>::type>::value ||
			detail::has_serialize<T, boost::is_class<T>::value>::value
		>, archive<write> &>::type operator<< (archive<write> & ar, detail::pvp<T, U> const & v) {
			std::string c = ar.get_context();
			ar.set_context(ar.compute_path(v.get_path()));
			v.serialize(ar);
			ar.set_context(c);
			return ar;
		}
		template <typename T, detail::pvp_type U> typename boost::disable_if<boost::mpl::bool_<
			boost::is_base_of<serializable, typename boost::remove_reference<typename boost::remove_cv<T>::type>::type>::value ||
			detail::has_serialize<T, boost::is_class<T>::value>::value
		>, archive<write> &>::type operator<< (archive<write> & ar, detail::pvp<T, U> const & v) {
			v.set_value(ar);
			return ar;
		}
		template <typename T> archive<read> & operator>> (archive<read> & ar, detail::pvp<T &, detail::plain> const & v) {
//			ar.get_data(v.get_path(ar.get_context()), v.get_value());
//			return ar;
		}
	}
	template <typename T> hdf5::detail::pvp<T &, hdf5::detail::plain> make_pvp(std::string const & p, T & v) {
		return hdf5::detail::pvp<T &, hdf5::detail::plain>(p, v);
	}
	template <typename T> hdf5::detail::pvp<T, hdf5::detail::ptr> make_pvp(std::string const & p, T * v) {
		return hdf5::detail::pvp<T, hdf5::detail::ptr>(p, v);
	}
	template <typename T> hdf5::detail::pvp<T, hdf5::detail::plain> make_pvp(std::string const & p, T const & v) {
		return hdf5::detail::pvp<T, hdf5::detail::plain>(p, v);
	}
	template <typename T> hdf5::detail::pvp<T, hdf5::detail::array> make_pvp(std::string const & p, T * v, std::size_t s) {
		return hdf5::detail::pvp<T, hdf5::detail::array>(p, v);
	}
}
#endif
