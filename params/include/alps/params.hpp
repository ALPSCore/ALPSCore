/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_PARAMS_HPP_INCLUDED_00f672a032d949a7aa0e760a6b6f0602
#define ALPS_PARAMS_HPP_INCLUDED_00f672a032d949a7aa0e760a6b6f0602

#include "alps/config.hpp"
#include "alps/hdf5/archive.hpp"
#include <alps/utilities/deprecated.hpp>


#ifdef ALPS_HAVE_MPI
#include <alps/utilities/mpi.hpp>
#include <alps/utilities/mpi_pair.hpp>
#endif

#include <map>
#include <iosfwd>

#include "./dictionary.hpp"

namespace alps {
    namespace params_ns {

        namespace detail {
            /// Type-description functor: returns a "typestring" given a type
            struct make_typestr : public boost::static_visitor<std::string> {
                template <typename T>
                std::string operator()(const T&) const { return apply<T>(); }

                template <typename T>
                static std::string apply() { return typeid(T).name(); } // FIXME: use pretty-typename instead?
            };

            /// Type-description pair (essentially, an std::pair with named fields)
            struct td_pair {
                std::pair<std::string,std::string> super_;
                /// Access typestring
                std::string& typestr() { return super_.first; }
                /// Access typestring
                const std::string& typestr() const { return super_.first; }
                /// Access description
                std::string& descr() { return super_.second; }
                /// Access description
                const std::string& descr() const { return super_.second; }
                /// Construct from typestring and description
                td_pair(const std::string& t, const std::string& d) : super_(t,d) {}
                /// Construct from description given a type
                template <typename T>
                static td_pair make_pair(const std::string& d) { return td_pair(make_typestr::apply<T>(), d); }
                /// Empty-pair ctor needed for MPI
                td_pair() : super_() {}
                /// Comparison
                bool operator==(const td_pair& rhs) const { return this->super_ == rhs.super_; }
// #ifdef ALPS_HAVE_MPI
//                 /// Broadcast
//                 friend void broadcast(const alps& comm, td_pair& val, int root) {
//                     broadcast(comm, val.super_, root);
//                 }
// #endif
            };
        } // detail::


        /// Parse sectioned INI file or HDF5 or command line, provide the results as dictionary.
        /**
           1. Default-constructed `params` object cannot be re-associated with a file;
              therefore, is 100% equivalent to `dictionary` ("is-a" dictionary).

           2. Lexing of the file and of the command line occurs at construction.
              Command line overrides the file. INI file name is taken from the command line.

           3. Parsing of a specific parameter occurs at the time of its type definition.
              There is no way for parameters to appear after the file and cmdline are read.

         */
        class params : public dictionary {
          private:
            typedef std::map<std::string,std::string> strmap;
            // typedef std::vector<std::string> strvec;


            typedef std::map<std::string, detail::td_pair> td_map_type;


            strmap raw_kv_content_;
            td_map_type td_map_;
            int err_status_;
            std::string argv0_;
            std::string archive_name_;

            void read_ini_file_(const std::string& inifile);
            void initialize_(int argc, const char* const* argv, const char* hdf5_path);

            template <typename T>
            bool assign_to_name_(const std::string& name, const std::string& strval);

            /// Does the job of define(), returns false if the name is missing in raw_argsand default must be checked
            template <typename T>
            bool define_(const std::string& name, const std::string& descr);

          public:
            /// Default ctor
            params() : dictionary(), raw_kv_content_(), td_map_(), err_status_(0), argv0_(), archive_name_() {}

            params(const std::string& inifile)
                : dictionary(), raw_kv_content_(), td_map_(), err_status_(0), argv0_(), archive_name_()
            {
                read_ini_file_(inifile);
            }

            params(int argc, const char* const* argv, const char* hdf5_path="/parameters")
                : dictionary(),
                  raw_kv_content_(),
                  td_map_(),
                  err_status_(),
                  argv0_(),
                  archive_name_()
            { initialize_(argc, argv, hdf5_path); }


            /// Convenience method: returns the "origin name"
            /** @Returns (parameter_file_name || restart_file name || program_name || "") **/
            std::string get_origin_name() const { return argv0_; }

            /// Conveninece method: true if the object was restored from an archive
            bool is_restored() const { return !archive_name_.empty(); }
            
            /// Convenience method: returns the archive name the object has been restored from, or throws
            std::string get_archive_name() const;
            
            /// No-errors status
            bool ok() const { return 0==err_status_; }

            /// Returns true if the objects are identical
            bool operator==(const params& rhs) const;

            /// Check whether a parameter was ever defined
            // FIXME: we don't really need it, must be removed from client code
            bool defined(const std::string& name) ALPS_DEPRECATED { return td_map_.count(name)!=0 || exists(name); }

            /// Defines a parameter; returns false on error, and records the error in the object
            template<typename T>
            params& define(const std::string& name, const std::string& descr);

            /// Defines a parameter with a default; returns false on error, and records the error in the object
            template<typename T>
            params& define(const std::string& name, const T& defval, const std::string& descr);

            /// Returns a string describing the parameter (or an empty string)
            const std::string get_descr(const std::string& name) const;

            friend void swap(params& p1, params& p2);

            /// Saves parameter object to an archive
            void save(alps::hdf5::archive&) const;

            /// Loads parameter object form an archive
            void load(alps::hdf5::archive&);

            /// Prints parameters to a stream in an unspecified format
            friend
            std::ostream& operator<<(std::ostream&, const params&);

#ifdef ALPS_HAVE_MPI
            // FIXME: should it be virtual?
            void broadcast(const alps::mpi::communicator& comm, int root);

            /// Broadcasting ctor. Reads file/params on root, broadcasts to everyone
            params(int argc, const char* const* argv, const alps::mpi::communicator& comm, int root, const char* hdf5_path="/parameters")
                : dictionary(),
                  raw_kv_content_(),
                  td_map_(),
                  err_status_(),
                  argv0_()

            {
                initialize_(argc,argv,hdf5_path);
                broadcast(comm, root);
            }
#endif
        };

    } // params_ns::
    typedef params_ns::params params;

#ifdef ALPS_HAVE_MPI
    namespace mpi {

        inline void broadcast(const alps::mpi::communicator& comm, alps::params_ns::detail::td_pair& tdp, int root) {
            broadcast(comm, tdp.super_, root);
        }

        inline void broadcast(const alps::mpi::communicator &comm, alps::params_ns::dictionary& dict, int root) {
            dict.broadcast(comm, root);
        }

        inline void broadcast(const alps::mpi::communicator &comm, alps::params_ns::params& p, int root) {
            p.broadcast(comm, root);
        }
    } // mpi::
#endif

} // alps::

#include "./params/params_impl.hpp"


#endif /* ALPS_PARAMS_HPP_INCLUDED_00f672a032d949a7aa0e760a6b6f0602 */
