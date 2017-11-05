/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params.cpp
    Contains implementation of alps::params */

#include <alps/params/iniparser_interface.hpp>
#include <alps/params.hpp>
#include <algorithm>
#include <sstream>

#include <alps/testing/unique_file.hpp> // FIXME!!! Temporary!
#include <fstream> // FIXME!!! Temporary!

#include <alps/hdf5/map.hpp>
#include <alps/hdf5/vector.hpp> // DEBUG! 

#include <boost/foreach.hpp>

#ifdef ALPS_HAVE_MPI
#include <alps/utilities/mpi_map.hpp>
#endif


namespace alps {
    namespace params_ns {

        void params::initialize_(int argc, const char* const * argv)
        {
            // shortcuts:
            typedef std::string::size_type size_type;
            const size_type& npos=std::string::npos;
            using std::string;
            
            if (argc==0) return;
            argv0_.assign(argv[0]);
            if (argc<2) return;
            
            std::vector<string> all_args(argv+1,argv+argc);
            std::stringstream cmd_options;
            bool file_args_mode=false;
            BOOST_FOREACH(const string& arg, all_args) {
                if (file_args_mode) {
                    read_ini_file_(arg);
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
                    read_ini_file_(arg);
                    continue;
                }
                if (npos==key_end) {
                    cmd_options << arg.substr(key_begin) << "=true\n";
                } else {
                    cmd_options << arg.substr(key_begin) << "\n";
                }
            }

            // FIXME!!!
            // This is very inefficient and is done only for testing.
            alps::testing::unique_file tmpfile(argv0_+".param.ini", alps::testing::unique_file::KEEP_AFTER);
            std::ofstream tmpstream(tmpfile.name().c_str());
            tmpstream << cmd_options.rdbuf();
            tmpstream.close();
            read_ini_file_(tmpfile.name());
            
        }
        
        void params::read_ini_file_(const std::string& inifile)
        {
            detail::iniparser parser(inifile);
            BOOST_FOREACH(const detail::iniparser::kv_pair& kv, parser()) {
                // FIXME!!! Check for duplicates and optionally warn!
                std::string key=kv.first;
                if (!key.empty() && key[0]==':') key.erase(0,1);
                raw_kv_content_[key]=kv.second;
            }
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
                (lhs.argv0_ == rhs.argv0_) &&
                (lhs_dict==rhs_dict);
        }


        void params::save(alps::hdf5::archive& ar) const {
            dictionary::save(ar);
            const std::string context=ar.get_context();
            // Convert the inifile map to vectors of keys, values
            std::vector<std::string> raw_keys, raw_vals;
            raw_keys.reserve(raw_kv_content_.size());
            raw_vals.reserve(raw_kv_content_.size());
            BOOST_FOREACH(const strmap::value_type& kv, raw_kv_content_) {
                raw_keys.push_back(kv.first);
                raw_vals.push_back(kv.second);
            }
            ar[context+"@ini_keys"] << raw_keys;
            ar[context+"@ini_values"] << raw_vals;
            ar[context+"@status"] << err_status_;
            ar[context+"@argv0"]  << argv0_;
            
            std::vector<std::string> keys=ar.list_children(context);
            BOOST_FOREACH(const std::string& key, keys) {
                td_map_type::const_iterator it=td_map_.find(key);
                
                if (it!=td_map_.end()) {
                    ar[key+"@description"] << it->second.descr();
                }
            }
        }
        
        void params::load(alps::hdf5::archive& ar) {
            params newpar;
            newpar.dictionary::load(ar);

            const std::string context=ar.get_context();

            ar[context+"@status"] >> err_status_;
            ar[context+"@argv0"]  >> argv0_;
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
            BOOST_FOREACH(const std::string& key, keys) {
                const std::string attr=key+"@description";
                if (ar.is_attribute(attr)) {
                    std::string descr;
                    ar[attr] >> descr;

                    const_iterator it=newpar.find(key);
                    if (newpar.end()==it) {
                        throw std::logic_error("params::load(): loading the dictionary"
                                               " missed key '"+key+"'??");
                    }
                    std::string typestr=apply_visitor(detail::make_typestr(), it);
                    newpar.td_map_.insert(std::make_pair(key, detail::td_pair(typestr, descr)));
                }
            }
            
            using std::swap;
            swap(*this, newpar);
        }


        std::ostream& operator<<(std::ostream& s, const params& p) {
            s << "[alps::params]"
              << " argv0='" << p.argv0_ << "' status=" << p.err_status_
              << "\nRaw kv:\n";
            BOOST_FOREACH(const params::strmap::value_type& kv, p.raw_kv_content_) {
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
                      << "' typestring='" << tdit->second.typestr() << "'";
                }
                s << std::endl;
            }
            return s;
        }


#ifdef ALPS_HAVE_MPI
        void params::broadcast(const alps::mpi::communicator& comm, int rank) {
            this->dictionary::broadcast(comm, rank);
            using alps::mpi::broadcast;
            broadcast(comm, raw_kv_content_, rank);
            broadcast(comm, td_map_, rank);
            broadcast(comm, err_status_, rank);
            broadcast(comm, argv0_, rank);
        }
#endif


        
    } // ::params_ns
}// alps::
            
