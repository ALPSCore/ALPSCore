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

#include <boost/foreach.hpp>

#ifdef ALPS_HAVE_MPI
#include <alps/utilities/mpi_map.hpp>
#include <alps/params/mpi_variant.hpp>
#endif

namespace alps {
    namespace params_ns {

#ifdef ALPS_HAVE_MPI
        void dict_value::broadcast(const alps::mpi::communicator& comm, int root)
        {
            using alps::mpi::broadcast;
            broadcast(comm, name_, root);
            broadcast<detail::dict_all_types>(comm, val_, root);
        }
#endif

        
        namespace {
            
            template <typename M>
            struct compare {
                typedef typename M::value_type pair_type;
                bool operator()(const pair_type& lhs, const pair_type& rhs) const
                {
                    return (lhs.first==rhs.first) && lhs.second.equals(rhs.second);
                }
            };

        }
        
        bool dictionary::equals(const dictionary &rhs) const 
        {
            if (this->size()!=rhs.size()) return false;
            return std::equal(map_.begin(), map_.end(), rhs.map_.begin(), compare<map_type>());
        }

#ifdef ALPS_HAVE_MPI
        // Defined here to avoid including <mpi_map.hpp> inside user header
        void dictionary::broadcast(const alps::mpi::communicator& comm, int root) { 
            using alps::mpi::broadcast;
            broadcast(comm, map_, root);
        }
#endif


        
        params::params(int argc, const char* const * argv)
            : dictionary(),
              raw_kv_content_(),
              td_map_(),
              err_status_(),
              argv0_()
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
                (lhs_dict==rhs_dict);
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
            
