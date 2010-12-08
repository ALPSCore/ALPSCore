/*****************************************************************************
 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations
 *
 * ALPS Libraries
 *
 * Copyright (C) 2008-2018 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
 *                            Lukas Gamper <gamperl -at- gmail.com>
 *
 * This software is part of the ALPS libraries, published under the ALPS
 * Library License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 * 
 * You should have received a copy of the ALPS Library License along with
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#include <alps/hdf5.hpp>
#include <boost/filesystem.hpp>
namespace alps {
  namespace hdf5 {
    namespace detail {
      
      herr_t error::callback(unsigned n, H5E_error2_t const * desc, void * buffer) 
      {
        *reinterpret_cast<std::ostringstream *>(buffer) 
            << "    #" 
            << convert<std::string>(n) 
            << " " << desc->file_name 
            << " line " 
            << convert<std::string>(desc->line) 
            << " in " 
            << desc->func_name 
            << "(): " 
            << desc->desc 
            << std::endl;
        return 0;
      }
      
      std::string error::invoke(hid_t id) 
      {
        std::ostringstream buffer;
        buffer << "HDF5 error: " << convert<std::string>(id) << std::endl;
        H5Ewalk2(H5E_DEFAULT, H5E_WALK_DOWNWARD, callback, &buffer);
        return buffer.str();
      }
      
     
      context::context(std::string const & filename, hid_t file_id, bool compress)
        : _compress(compress)
        , _revision(0)
        , _state_id(-1)
        , _log_id(-1)
        , _filename(filename)
        , _file_id(file_id)
        {
            if (_compress) {
                unsigned int flag;
                detail::check_error(H5Zget_filter_info(H5Z_FILTER_SZIP, &flag));
                _compress = flag & H5Z_FILTER_CONFIG_ENCODE_ENABLED;
            }
        }
        
      context::~context() 
      {
          try {
              H5Fflush(_file_id, H5F_SCOPE_GLOBAL);
              if (_state_id > -1)
                  detail::check_type(_state_id);
              if (_log_id > -1)
                  detail::check_type(_log_id);
#ifndef ALPS_HDF5_CLOSE_GREEDY
              if (
                  H5Fget_obj_count(_file_id, H5F_OBJ_DATATYPE) > (_state_id == -1 ? 0 : 1) + (_log_id == -1 ? 0 : 1)
                  || H5Fget_obj_count(_file_id, H5F_OBJ_ALL) - H5Fget_obj_count(_file_id, H5F_OBJ_FILE) - H5Fget_obj_count(_file_id, H5F_OBJ_DATATYPE) > 0
                  ) {
                  std::cerr << "Not all resources closed in file '" << _filename << "'" << std::endl;
                  std::abort();
              }
#endif
          } catch (std::exception & ex) {
              std::cerr << "Error destructing HDF5 context of file '" << _filename << "'\n" << ex.what() << std::endl;
              std::abort();
          }
      }
        
        hid_t creator::open_reading(std::string const & file) {
            if (!boost::filesystem::exists(file))
                ALPS_HDF5_THROW_RUNTIME_ERROR("file does not exists: " + file)
                if (detail::check_error(H5Fis_hdf5(file.c_str())) == 0)
                    ALPS_HDF5_THROW_RUNTIME_ERROR("no valid hdf5 file: " + file)
                    return H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        }
        
        hid_t creator::open_writing(std::string const & file) {
            hid_t file_id = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
            return file_id < 0 ? H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT) : file_id;
        }
        
    } // namespace detail
      
      
      
      archive::archive(archive const & rhs)
      : _path_context(rhs._path_context)
      , _context(rhs._context)
      {}
      
    archive::~archive() 
      {
          try {
              H5Fflush(file_id(), H5F_SCOPE_GLOBAL);
          } catch (std::exception & ex) {
              std::cerr << "Error destructing archive of file '" << filename() << "'\n" << ex.what() << std::endl;
              std::abort();
          }
      }
      
      std::string const & archive::filename() const 
      {
          return _context->_filename;
      }
      
      std::string archive::encode_segment(std::string const & s) 
      {
          std::string r = s;
          char chars[] = {'&', '/'};
          for (std::size_t i = 0; i < sizeof(chars); ++i)
              for (std::size_t pos = r.find_first_of(chars[i]); pos < std::string::npos; pos = r.find_first_of(chars[i], pos + 1))
                  r = r.substr(0, pos) + "&#" + detail::convert<std::string>(static_cast<int>(chars[i])) + ";" + r.substr(pos + 1);
          return r;
      }
      
      std::string archive::decode_segment(std::string const & s) 
      {
          std::string r = s;
          for (std::size_t pos = r.find_first_of('&'); pos < std::string::npos; pos = r.find_first_of('&', pos + 1))
              r = r.substr(0, pos) + static_cast<char>(detail::convert<int>(r.substr(pos + 2, r.find_first_of(';', pos) - pos - 2))) + r.substr(r.find_first_of(';', pos) + 1);
          return r;
      }
      
      void archive::commit(std::string const & name) 
      {
          set_attribute("/revisions/@last", ++_context->_revision);
          set_group("/revisions/" + detail::convert<std::string>(revision()));
          std::string time = boost::posix_time::to_iso_string(boost::posix_time::second_clock::local_time());
          detail::internal_log_type v = {
              std::strcpy(new char[time.size() + 1], time.c_str()),
              std::strcpy(new char[name.size() + 1], name.c_str())
          };
          set_attribute("/revisions/" + detail::convert<std::string>(revision()) + "/@info", v);
          delete[] v.time;
          delete[] v.name;
      }
      
      std::vector<std::pair<std::string, std::string> > archive::list_revisions() const 
      {
          // TODO: implement
          return std::vector<std::pair<std::string, std::string> >();
      }
      
      void archive::export_revision(std::size_t revision, std::string const & file) const 
      {
          // TODO: implement
      }
      
      std::string archive::get_context() const 
      {
          return _path_context;
      }
      
      void archive::set_context(std::string const & context) 
      {
          _path_context = context;
      }
      
      std::string archive::complete_path(std::string const & p) const 
      {
          if (p.size() && p[0] == '/')
              return p;
          else if (p.size() < 2 || p.substr(0, 2) != "..")
              return _path_context + (_path_context.size() == 1 || !p.size() ? "" : "/") + p;
          else {
              std::string s = _path_context;
              std::size_t i = 0;
              for (; s.size() && p.substr(i, 2) == ".."; i += 3)
                  s = s.substr(0, s.find_last_of('/'));
              return s + (s.size() == 1 || !p.substr(i).size() ? "" : "/") + p.substr(i);
          }
      }
      
      bool archive::is_group(std::string const & p) const 
      {
          try {
              hid_t id = H5Gopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT);
              return id < 0 ? false : detail::check_group(id) != 0;
          } catch (std::exception & ex) {
              ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
          }
      }
      
      bool archive::is_data(std::string const & p) const 
      {
          try {
              hid_t id = H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT);
              return id < 0 ? false : detail::check_data(id) != 0;
          } catch (std::exception & ex) {
              ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
          }
      }
      
      bool archive::is_attribute(std::string const & p) const 
      {
          try {
              if (p.find_last_of('@') == std::string::npos)
                  ALPS_HDF5_THROW_RUNTIME_ERROR("no attribute path: " + complete_path(p))
                  hid_t parent_id;
              if (is_group(complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1)))
                  parent_id = detail::check_error(H5Gopen2(file_id(), complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1).c_str(), H5P_DEFAULT));
              else if (is_data(complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1)))
                  parent_id = detail::check_error(H5Dopen2(file_id(), complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1).c_str(), H5P_DEFAULT));
              else
#ifdef ALPS_HDF5_READ_GREEDY
                  return false;
#else
              ALPS_HDF5_THROW_RUNTIME_ERROR("unknown path: " + complete_path(p))
#endif
              bool exists = detail::check_error(H5Aexists(parent_id, p.substr(p.find_last_of('@') + 1).c_str()));
              if (is_group(complete_path(p).substr(0, complete_path(p).find_last_of('@') - 1)))
                  detail::check_group(parent_id);
              else
                  detail::check_data(parent_id);
              return exists;
          } catch (std::exception & ex) {
              ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
          }
      }
      
      std::vector<std::size_t> archive::extent(std::string const & p) const 
      {
          try {
              if (is_null(p))
                  return std::vector<std::size_t>(1, 0);
              else if (is_scalar(p))
                  return std::vector<std::size_t>(1, 1);
              std::vector<hsize_t> buffer(dimensions(p), 0);
              hid_t space_id;
              if (p.find_last_of('@') != std::string::npos) {
                  detail::attribute_type attr_id(open_attribute(complete_path(p)));
                  space_id = H5Aget_space(attr_id);
              } else {
                  detail::data_type data_id(H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                  space_id = H5Dget_space(data_id);
              }
              detail::check_error(H5Sget_simple_extent_dims(space_id, &buffer.front(), NULL));
              detail::check_space(space_id);
              std::vector<std::size_t> extent(buffer.size(), 0);
              std::copy(buffer.begin(), buffer.end(), extent.begin());
              if (is_data(p) && is_attribute(p + "/@__complex__") && extent.back() == 2) {
                  extent.pop_back();
                  if (!extent.size())
                      return std::vector<std::size_t>(1, 1);
              }
              return extent;
          } catch (std::exception & ex) {
              ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
          }
      }
      
      std::size_t archive::dimensions(std::string const & p) const 
      {
          try {
              if (p.find_last_of('@') != std::string::npos) {
                  detail::attribute_type attr_id(open_attribute(complete_path(p)));
                  return static_cast<hid_t>(detail::check_error(H5Sget_simple_extent_dims(detail::space_type(H5Aget_space(attr_id)), NULL, NULL)));
              } else {
                  detail::data_type data_id(H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                  return static_cast<hid_t>(detail::check_error(H5Sget_simple_extent_dims(detail::space_type(H5Dget_space(data_id)), NULL, NULL)));
              }
          } catch (std::exception & ex) {
              ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
          }
      }
      
      bool archive::is_scalar(std::string const & p) const 
      {
          try {
              hid_t space_id;
              if (p.find_last_of('@') != std::string::npos) {
                  detail::attribute_type attr_id(open_attribute(complete_path(p)));
                  space_id = H5Aget_space(attr_id);
              } else {
                  detail::data_type data_id(H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                  space_id = H5Dget_space(data_id);
              }
              H5S_class_t type = H5Sget_simple_extent_type(space_id);
              detail::check_space(space_id);
              if (type == H5S_NO_CLASS)
                  ALPS_HDF5_THROW_RUNTIME_ERROR("error reading class " + complete_path(p))
                  return type == H5S_SCALAR;
          } catch (std::exception & ex) {
              ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
          }
      }
      
      bool archive::is_string(std::string const & p) const 
      {
          try {
              hid_t type_id;
              if (p.find_last_of('@') != std::string::npos) {
                  detail::attribute_type attr_id(open_attribute(complete_path(p)));
                  type_id = H5Aget_type(attr_id);
              } else {
                  detail::data_type data_id(H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                  type_id = H5Dget_type(data_id);
              }
              detail::type_type native_id(H5Tget_native_type(type_id, H5T_DIR_ASCEND));
              detail::check_type(type_id);
              return H5Tget_class(native_id) == H5T_STRING;
          } catch (std::exception & ex) {
              ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
          }
      }
      
      bool archive::is_int(std::string const & p) const 
      {
          return is_type<int>(p);
      }
      
      bool archive::is_long(std::string const & p) const 
      {
          return is_type<long>(p);
      }
      
      bool archive::is_float(std::string const & p) const 
      {
          return is_type<float>(p);
      }
      
      bool archive::is_double(std::string const & p) const 
      {
          return is_type<double>(p);
      }
      
      bool archive::is_complex(std::string const & p) const 
      {
          // TODO: implement!
          ALPS_HDF5_THROW_RUNTIME_ERROR("not impl");
      }
      
      bool archive::is_null(std::string const & p) const 
      {
          try {
              hid_t space_id;
              if (p.find_last_of('@') != std::string::npos) {
                  detail::attribute_type attr_id(open_attribute(complete_path(p)));
                  space_id = H5Aget_space(attr_id);
              } else {
                  detail::data_type data_id(H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                  space_id = H5Dget_space(data_id);
              }
              H5S_class_t type = H5Sget_simple_extent_type(space_id);
              detail::check_space(space_id);
              if (type == H5S_NO_CLASS)
                  ALPS_HDF5_THROW_RUNTIME_ERROR("error reading class " + complete_path(p))
                  return type == H5S_NULL;
          } catch (std::exception & ex) {
              ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
          }
      }
      
      void archive::serialize(std::string const & p) 
      {
          if (p.find_last_of('@') != std::string::npos)
              ALPS_HDF5_THROW_RUNTIME_ERROR("attributes needs to be a scalar type or a string" + p)
              else
                  set_group(complete_path(p));
      }
      
      void archive::delete_data(std::string const & p) const 
      {
          try {
              if (is_data(p))
                  // TODO: implement provenance
                  detail::check_error(H5Ldelete(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
              else
                  ALPS_HDF5_THROW_RUNTIME_ERROR("the path does not exists: " + p)
                  } catch (std::exception & ex) {
                      ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                  }
      }
      
      void archive::delete_group(std::string const & p) const 
      {
          try {
              if (is_group(p))
                  // TODO: implement provenance
                  detail::check_error(H5Ldelete(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
              else
                  ALPS_HDF5_THROW_RUNTIME_ERROR("the path does not exists: " + p)
                  } catch (std::exception & ex) {
                      ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
                  }
      }
      
      void archive::delete_attribute(std::string const & p) const 
      {
          try {
              // TODO: implement
          } catch (std::exception & ex) {
              ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
          }
      }
      
      std::vector<std::string> archive::list_children(std::string const & p) const 
      {
          try {
              std::vector<std::string> list;
              if (!is_group(p))
                  ALPS_HDF5_THROW_RUNTIME_ERROR("The group '" + p + "' does not exists.")
                  detail::group_type group_id(H5Gopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
              detail::check_error(H5Literate(group_id, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, child_visitor, reinterpret_cast<void *>(&list)));
              return list;
          } catch (std::exception & ex) {
              ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
          }
      }
      
      std::vector<std::string> archive::list_attributes(std::string const & p) const 
      {
          try {
              std::vector<std::string> list;
              if (is_group(p)) {
                  detail::group_type id(H5Gopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                  detail::check_error(H5Aiterate2(id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, attr_visitor, reinterpret_cast<void *>(&list)));
              } else if(is_data(p)) {
                  detail::data_type id(H5Dopen2(file_id(), complete_path(p).c_str(), H5P_DEFAULT));
                  detail::check_error(H5Aiterate2(id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, NULL, attr_visitor, reinterpret_cast<void *>(&list)));
              } else
                  ALPS_HDF5_THROW_RUNTIME_ERROR("The path '" + p + "' does not exists.")
                  return list;
          } catch (std::exception & ex) {
              ALPS_HDF5_THROW_RUNTIME_ERROR("file: " + filename() + ", path: " + p + "\n" + ex.what());
          }
      }
      
    void oarchive::serialize(std::string const & p) 
    {
        if (p.find_last_of('@') != std::string::npos)
            ALPS_HDF5_THROW_RUNTIME_ERROR("attributes needs to be a scalar type or a string" + p)
        else
            set_group(complete_path(p));
    }

    std::map<std::pair<std::string, bool>, boost::weak_ptr<detail::context> > archive::_pool;

  }    
}
