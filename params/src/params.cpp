/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params.cpp
    Contains implementation of alps::params */

#include <alps/params/iniparser_interface.hpp>
#include <alps/params.hpp>
#include <algorithm>
#include <sstream>
#include <iomanip> // for help pretty-printing
#include <iterator> // for ostream_iterator
#include <cstring> // for memcmp()
#include <boost/optional.hpp>

#include <alps/utilities/fs/get_basename.hpp>

#include <alps/testing/fp_compare.hpp>

#include <alps/utilities/temporary_filename.hpp> // FIXME!!! Temporary!
#include <fstream>

#include <alps/hdf5/map.hpp>
#include <alps/hdf5/vector.hpp>


#ifdef ALPS_HAVE_MPI
#include <alps/utilities/mpi_map.hpp>
#include <alps/utilities/mpi_vector.hpp>
#endif


namespace alps {
    namespace params_ns {

        namespace {
            // Helper function to try to open an HDF5 archive, return "none" if it fails
            boost::optional<alps::hdf5::archive> try_open_ar(const std::string& fname, const char* mode)
            {
                try {
                    //read in hdf5 checksum of file and verify it's a hdf5 file
                    {
                        std::ifstream f(fname.c_str(),std::ios::binary);
                        if(!f.good()) return boost::none;
                        static const char hdf5_checksum[]={char(137),72,68,70,13,10,26,10};
                        char firstbytes[sizeof(hdf5_checksum)];
                        f.read(firstbytes, sizeof(firstbytes));
                        if(!f.good() || memcmp(hdf5_checksum, firstbytes, sizeof(firstbytes))!=0) return boost::none;
                    }
                    return alps::hdf5::archive(fname, mode);
                } catch (alps::hdf5::archive_error& ) {
                    return boost::none;
                }
            };


            // Helper function to read INI file into the provided map
            template <typename MAP_T>
            void ini_file_to_map(const std::string& ini_name, MAP_T& map)
            {
                detail::iniparser parser(ini_name);
                for(const detail::iniparser::kv_pair& kv: parser()) {
                    // FIXME!!! Check for duplicates and optionally warn!
                    std::string key=kv.first;
                    if (!key.empty() && key[0]=='.') key.erase(0,1);
                    map[key]=kv.second;
                }
            }

            // Helper functor to convert a string to a value of the
            // same type as a dict_value and assign it to the dict_value
            class parse_visitor {
                const std::string& strparam_val_;
                dictionary::value_type& dictval_;
              public:
                typedef bool result_type;

                parse_visitor(const std::string& strparam_val, dictionary::value_type& dval):
                    strparam_val_(strparam_val), dictval_(dval)
                {}

                template <typename BOUND_T>
                result_type operator()(const BOUND_T& bound_val) const
                {
                    boost::optional<BOUND_T> maybe_val=detail::parse_string<BOUND_T>::apply(strparam_val_);
                    if (!maybe_val) return false;
                    dictval_=*maybe_val;
                    return true;
                }

                result_type operator()(const dictionary::value_type::None&) const
                {
                    return true;
                }
            };


            /// Default help description
            static const char Default_help_description[]="Print help message";
        }

        params::params(const std::string& inifile)
            : dictionary(), raw_kv_content_(), td_map_(), err_status_(), origins_(), help_header_()
        {
            read_ini_file_(inifile);
            if (!defined("help")) define("help", Default_help_description);
        }



        params::params(int argc, const char* const* argv, const char* hdf5_path)
            : dictionary(),
              raw_kv_content_(),
              td_map_(),
              err_status_(),
              origins_(),
              help_header_()
        {
            initialize_(argc, argv, hdf5_path);
            if (!defined("help")) define("help", Default_help_description);
        }


        int params::get_ini_name_count() const
        {
            return origins_.data().size()-origins_type::INIFILES;
        }

        std::string params::get_ini_name(int n) const
        {
            if (n<0 || n>=get_ini_name_count()) return std::string();
            return origins_.data()[origins_type::INIFILES+n];
        }

        std::string params::get_argv0() const
        {
            return origins_.data()[origins_type::ARGV0];
        }

        std::string params::get_archive_name() const
        {
            if (!this->is_restored()) throw std::runtime_error("The parameters object is not restored from an archive");
            return origins_.data()[origins_type::ARCHNAME];
        }

        params& params::description(const std::string &message)
        {
            help_header_=message;
            if (!defined("help")) define("help", Default_help_description);
            return *this;
        }

        bool params::has_unused_(std::ostream& out, const std::string* prefix_ptr) const
        {
            strvec unused;
            for(const strmap::value_type& kv: raw_kv_content_) {
                bool relevant = !prefix_ptr  // no specific prefix?
                                || (prefix_ptr->empty() ? kv.first.find('.')==std::string::npos // top-level section?
                                                        : kv.first.find(*prefix_ptr+".")==0);   // starts with sec name?
                if (relevant && !this->exists(kv.first)) {
                    unused.push_back(kv.first+" = "+kv.second);
                }
            }
            if (!unused.empty()) {
                out << "The following arguments are supplied, but never referenced:\n";
                std::copy(unused.begin(), unused.end(), std::ostream_iterator<std::string>(out,"\n"));
            }
            return !unused.empty();
        }

        bool params::has_unused(std::ostream& out, const std::string& subsection) const
        {
            return has_unused_(out, &subsection);
        }

        bool params::has_unused(std::ostream& out) const
        {
            return has_unused_(out, 0);
        }

        std::ostream& params::print_help(std::ostream& out) const
        {
            out << help_header_ << "\nAvailable options:\n";

            typedef std::pair<std::string, std::string> nd_pair; // name and description
            typedef std::vector<nd_pair> nd_vector;
            nd_vector name_and_description;
            name_and_description.resize(td_map_.size());
            std::string::size_type names_column_width=0;

            // prepare 2 columns: parameters and their description
            for(const td_map_type::value_type& tdp: td_map_) {
                const std::string name_and_type=tdp.first + " (" +  tdp.second.typestr() + "):";
                if (names_column_width<name_and_type.size()) names_column_width=name_and_type.size();

                std::ostringstream ostr;
                boolalpha(ostr);
                ostr << tdp.second.descr();
                if (this->exists(tdp.first) && tdp.first!="help") {
                    ostr << " (default value: ";
                    print(ostr, (*this)[tdp.first], true) << ")";
                }
                // place the output on the line corresponding to definiton order
                int defnum=tdp.second.defnumber();
                if (defnum<0 || static_cast<unsigned int>(defnum)>=name_and_description.size()) {
                    std::ostringstream errmsg;
                    errmsg << "Invalid entry in parameters object.\n"
                           << "name='" << tdp.first
                           << "' defnumber=" << defnum;
                    throw std::logic_error(errmsg.str());
                }
                name_and_description[defnum]=std::make_pair(name_and_type, ostr.str());
            }

            // print the columns
            std::ostream::fmtflags oldfmt=out.flags();
            left(out);
            names_column_width += 4;
            for(const nd_pair& ndp: name_and_description) {
                out << std::left << std::setw(names_column_width) << ndp.first << ndp.second << "\n";
            }
            out.flags(oldfmt);

            return out;
        }

        bool params::help_requested() const
        {
            return this->exists<bool>("help") && (*this)["help"].as<bool>();
        }

        bool params::help_requested(std::ostream& out) const
        {
            if (!this->help_requested()) return false;
            print_help(out);
            return true;
        }

        bool params::has_missing(std::ostream& out) const
        {
            if (this->ok()) return false;
            std::copy(err_status_.begin(), err_status_.end(), std::ostream_iterator<std::string>(out,"\n"));
            return true;
        }

        void params::initialize_(int argc, const char* const * argv, const char* hdf5_path)
        {
            // shortcuts:
            typedef std::string::size_type size_type;
            const size_type& npos=std::string::npos;
            using std::string;
            using boost::optional;

            if (argc==0) return;
            origins_.data()[origins_type::ARGV0].assign(argv[0]);
            if (argc<2) return;

            std::vector<string> all_args(argv+1,argv+argc);
            std::stringstream cmd_options;
            std::vector<string> ini_files;
            optional<string> restored_from_archive;
            bool file_args_mode=false;
            for(const string& arg: all_args) {
                if (file_args_mode) {
                    ini_files.push_back(arg);
                    continue;
                }
                size_type key_end=arg.find('=');
                size_type key_begin=0;
                if (arg.substr(0,2)=="--") {
                    if (arg.size()==2) {
                        file_args_mode=true;
                        continue;
                    }
                    key_begin=2;
                } else if  (arg.substr(0,1)=="-") {
                    key_begin=1;
                }
                if (0==key_begin && npos==key_end) {
                    if (hdf5_path) {
                        optional<alps::hdf5::archive> maybe_ar=try_open_ar(arg, "r");
                        if (maybe_ar) {
                            if (restored_from_archive) {
                                throw archive_conflict("More than one archive is specified in command line",
                                                       *restored_from_archive, arg);
                            }
                            maybe_ar->set_context(hdf5_path);
                            this->load(*maybe_ar);
                            origins_.data()[origins_type::ARCHNAME]=arg;
                            restored_from_archive=arg;
                            continue;
                        }
                    }

                    ini_files.push_back(arg);
                    continue;
                }
                if (npos==key_end) {
                    cmd_options << arg.substr(key_begin) << "=true\n";
                } else {
                    cmd_options << arg.substr(key_begin) << "\n";
                }
            }
            for (auto fname: ini_files) {
                read_ini_file_(fname);
            }

            // FIXME!!!
            // This is very inefficient and is done only for testing.
            std::string tmpfile_name=alps::temporary_filename("tmp_ini_file");
            std::ofstream tmpstream(tmpfile_name.c_str());
            tmpstream << cmd_options.rdbuf();
            tmpstream.close();
            ini_file_to_map(tmpfile_name, raw_kv_content_);

            if (restored_from_archive) {
                // The parameter object was restored from archive, and
                // some key-values may have been supplied. We need to
                // go through the already `define<T>()`-ed map values
                // and try to parse the supplied string values as the
                // corresponding types.

                // It's a bit of a mess:
                // 1) We rely on that we can iterate over dictionary as a map
                // 2) Although it's const-iterator, we know that underlying map can be modified
                for (auto& kv: *this) {
                    const auto& key=kv.first;
                    auto raw_kv_it=raw_kv_content_.find(key);
                    if (raw_kv_it != raw_kv_content_.end()) {
                        bool ok=apply_visitor(parse_visitor(raw_kv_it->second, const_cast<dictionary::value_type&>(kv.second)), kv.second);
                        if (!ok) {
                            const auto typestr=td_map_[key].typestr();
                            throw exception::value_mismatch(key, "String '"+raw_kv_it->second+"' can't be parsed as type '"+typestr+"'");
                        }
                    }
                }
            }
        }

        void params::read_ini_file_(const std::string& inifile)
        {
            ini_file_to_map(inifile, raw_kv_content_);
            origins_.data().push_back(inifile);
        }

        const std::string params::get_descr(const std::string& name) const
        {
            td_map_type::const_iterator it=td_map_.find(name);
            return (td_map_.end()==it)? std::string() : it->second.descr();
        }

        bool params::operator==(const alps::params_ns::params& rhs) const
        {
            const params& lhs=*this;
            const dictionary& lhs_dict=*this;
            const dictionary& rhs_dict=rhs;
            return
                (lhs.raw_kv_content_ == rhs.raw_kv_content_) &&
                (lhs.td_map_ == rhs.td_map_) &&
                (lhs.err_status_ == rhs.err_status_) &&
                (lhs.help_header_ == rhs.help_header_) &&
                (lhs_dict==rhs_dict);
                /* origins_ is excluded from comparison */
        }


        void params::save(alps::hdf5::archive& ar) const {
            dictionary::save(ar);
            const std::string context=ar.get_context();
            // Convert the inifile map to vectors of keys, values
            std::vector<std::string> raw_keys, raw_vals;
            raw_keys.reserve(raw_kv_content_.size());
            raw_vals.reserve(raw_kv_content_.size());
            for(const strmap::value_type& kv: raw_kv_content_) {
                raw_keys.push_back(kv.first);
                raw_vals.push_back(kv.second);
            }
            ar[context+"@ini_keys"] << raw_keys;
            ar[context+"@ini_values"] << raw_vals;
            ar[context+"@status"] << err_status_;
            ar[context+"@origins"]  << origins_.data();
            ar[context+"@help_header"]  << help_header_;

            std::vector<std::string> keys=ar.list_children(context);
            for(const std::string& key: keys) {
                td_map_type::const_iterator it=td_map_.find(key);

                if (it!=td_map_.end()) {
                    ar[key+"@description"] << it->second.descr();
                    ar[key+"@defnumber"] << it->second.defnumber();
                }
            }
        }

        void params::load(alps::hdf5::archive& ar) {
            params newpar;
            newpar.dictionary::load(ar);

            const std::string context=ar.get_context();

            ar[context+"@status"] >> newpar.err_status_;
            ar[context+"@origins"]  >> newpar.origins_.data();
            newpar.origins_.check();
            ar[context+"@help_header"]  >> newpar.help_header_;

            // Get the vectors of keys, values and convert them back to a map
            {
                typedef std::vector<std::string> stringvec;
                stringvec raw_keys, raw_vals;
                ar[context+"@ini_keys"] >> raw_keys;
                ar[context+"@ini_values"] >> raw_vals;
                if (raw_keys.size()!=raw_vals.size()) {
                    throw std::invalid_argument("params::load(): invalid ini-file data in HDF5 (size mismatch)");
                }
                stringvec::const_iterator key_it=raw_keys.begin();
                stringvec::const_iterator val_it=raw_vals.begin();
                for (; key_it!=raw_keys.end(); ++key_it, ++val_it) {
                    strmap::const_iterator insloc=newpar.raw_kv_content_.insert(newpar.raw_kv_content_.end(), std::make_pair(*key_it, *val_it));
                    if (insloc->second!=*val_it) {
                        throw std::invalid_argument("params::load(): invalid ini-file data in HDF5 (repeated key '"+insloc->first+"')");
                    }
                }
            }
            std::vector<std::string> keys=ar.list_children(context);
            for(const std::string& key: keys) {
                const std::string d_attr=key+"@description";
                const std::string num_attr=key+"@defnumber";
                if (ar.is_attribute(d_attr)) {
                    std::string descr;
                    ar[d_attr] >> descr;

                    int dn=-1;
                    if (ar.is_attribute(num_attr)) {
                        ar[num_attr] >> dn;
                    } else {
                         // FIXME? Issue a warning instead? How?
                        throw std::runtime_error("Invalid HDF5 format: missing attribute "+ num_attr);
                    }

                    const_iterator it=newpar.find(key);
                    if (newpar.end()==it) {
                        throw std::logic_error("params::load(): loading the dictionary"
                                               " missed key '"+key+"'??");
                    }
                    std::string typestr=apply_visitor(detail::make_typestr(), it);
                    newpar.td_map_.insert(std::make_pair(key, detail::td_type(typestr, descr, dn)));
                }
            }

            using std::swap;
            swap(*this, newpar);
        }

        namespace {
            // Printing of a vector
            // FIXME!!! Consolidate with other definitions and move to alps::utilities
            template <typename T>
            inline std::ostream& operator<<(std::ostream& strm, const std::vector<T>& vec)
            {
                typedef std::vector<T> vtype;
                typedef typename vtype::const_iterator itype;

                strm << "[";
                itype it=vec.begin();
                const itype end=vec.end();

                if (end!=it) {
                    strm << *it;
                    for (++it; end!=it; ++it) {
                        strm << ", " << *it;
                    }
                }
                strm << "]";

                return strm;
            }
        }

        std::ostream& operator<<(std::ostream& s, const params& p) {
            s << "[alps::params]"
              << " origins=" << p.origins_.data() << " status=" << p.err_status_
              << "\nRaw kv:\n";
            for(const params::strmap::value_type& kv: p.raw_kv_content_) {
                s << kv.first << "=" << kv.second << "\n";
            }
            s << "[alps::params] Dictionary:\n";
            for (params::const_iterator it=p.begin(); it!=p.end(); ++it) {
                const std::string& key=it->first;
                const dict_value& val=it->second;
                s << key << " = " << val;
                params::td_map_type::const_iterator tdit = p.td_map_.find(key);
                if (tdit!=p.td_map_.end()) {
                    s << " descr='" << tdit->second.descr()
                      << "' typestring='" << tdit->second.typestr() << "'"
                      << "' defnum=" << tdit->second.defnumber();
                }
                s << std::endl;
            }
            return s;
        }

        std::string origin_name(const params& p)
        {
            std::string origin;
            if (p.is_restored()) origin=p.get_archive_name();
            else if (p.get_ini_name_count()>0) origin=p.get_ini_name(0);
            else origin=alps::fs::get_basename(p.get_argv0());
            return origin;
        }

#ifdef ALPS_HAVE_MPI
        void params::broadcast(const alps::mpi::communicator& comm, int rank) {
            this->dictionary::broadcast(comm, rank);
            using alps::mpi::broadcast;
            broadcast(comm, raw_kv_content_, rank);
            broadcast(comm, td_map_, rank);
            broadcast(comm, err_status_, rank);
            broadcast(comm, origins_.data(), rank);
            origins_.check();
        }
#endif
    } // ::params_ns
}// alps::
