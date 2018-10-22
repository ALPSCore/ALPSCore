/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
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
                static std::string apply() { return detail::type_info<T>::pretty_name(); }
            };

            /// Param_type with description
            struct td_type {
                std::string typestr_;
                std::string descr_;
                int defnumber_;
                /// Access typestring
                std::string& typestr() { return typestr_; }
                /// Access typestring
                const std::string& typestr() const { return typestr_; }
                /// Access description
                std::string& descr() { return descr_; }
                /// Access description
                const std::string& descr() const { return descr_; }
                /// Access defnumber
                int defnumber() const { return defnumber_; }
                /// Access defnumber
                int& defnumber() { return defnumber_; }
                /// Construct from typestring, description and number
                td_type(const std::string& t, const std::string& d, int n) : typestr_(t), descr_(d), defnumber_(n) {}
                /// Construct from description given a type and a number
                template <typename T>
                static td_type make_pair(const std::string& d, int n) { return td_type(make_typestr::apply<T>(), d, n); }
                /// Empty-pair ctor needed for MPI
                td_type() : typestr_(), descr_(), defnumber_(-1) {}
                /// Comparison
                bool operator==(const td_type& rhs) const {
                    return typestr_==rhs.typestr_ &&
                           descr_==rhs.descr_ &&
                           defnumber_==rhs.defnumber_;
                }
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
            typedef std::vector<std::string> strvec;

            // Small inner class to keep "origins" together
            struct origins_type {
                enum {
                    ARGV0=0,
                    ARCHNAME=1,
                    INIFILES=2
                };

                strvec data_;
                origins_type(): data_(INIFILES) {}
                strvec& data() { return data_; }
                const strvec& data() const { return data_; }
                void check() {
                    if (data_.size()<INIFILES)
                        throw std::logic_error("params::origins_type invariants violation");
                }
            };

            typedef std::map<std::string, detail::td_type> td_map_type;

            strmap raw_kv_content_;
            td_map_type td_map_;
            strvec err_status_;
            origins_type origins_;
            std::string help_header_;

            void read_ini_file_(const std::string& inifile);
            void initialize_(int argc, const char* const* argv, const char* hdf5_path);

            template <typename T>
            bool assign_to_name_(const std::string& name, const std::string& strval);

            bool has_unused_(std::ostream& out, const std::string* prefix_ptr) const;

            /// Does the job of define(), returns false if the name is missing in raw_argsand default must be checked
            template <typename T>
            bool define_(const std::string& name, const std::string& descr);

          public:
            /// Default ctor: creates an empty parameters object
            params() : dictionary(), raw_kv_content_(), td_map_(), err_status_(), origins_(), help_header_() {}

            /// Constructor from INI file
            /** Reads the provided file (in INI format). Automatically defines `--help` flag.

                @param inifile Path to the INI file
             **/
            params(const std::string& inifile);

            /// Constructor from command line and parameter files.
            /** Tries to see if the file is an HDF5, in which case restores the object from the
                HDF5 file, ignoring the command line. Automatically defines `--help` flag.

                @param argc Number of command line arguments (as in `main(int argc, char** argv)`)
                @param argv Array of pointers to command line arguments (as in `main(int argc, char** argv)`)
                @param hdf5_path path to HDF5 dataset containing the saved parameter object
                       (NULL if this functionality is not needed)


                The parameters are supplied in the `--key=value`
                format; the double dash (`--`) can be shortened to a single dash (`-`)
                or omitted.  The `=value` part may be omitted, in
                which case it is interpreted as `=false`, and the key
                must be preceded by at least one dash (`-`).

                If an argument is not recognized as a key-value pair
                (that is, it does not contain the `=` character and
                does not start with `-`) it is interpreted as a file
                name.

                If a double-dash argument (`--`) is encountered, it is
                skipped, and all arguments following it are
                interpreted as file names.

                The files are either HDF5-formatted archives or
                INI-formatted files. INI file can be sectoned by
                `[section]` headers: the key `some_key` in section
                `[some_section]` is interpreted as a key
                `some_section.some_key`.

                The command line is parsed in the following way:

                1. The whole command line is scanned.
                2. If an (HDF5) archive file name is present, the parameters object is loaded from the archive.
                3. If there is another archive file name, an exception of type
                   `alps::params::archive_conflict` is thrown.
                4. All INI files are read in the order given.
                5. The command line  arguments are read.
                    1. If a key is encountered more than once, the
                    latter occurence silently overrides the former. Thus,
                    the values supplied in the command line override values from INI files
                    and archives.
                    2. If a key was present and `define<T>()`d in the archive, the new value must be parsable
                    as type `T`; otherwise an exception of type
                    `alps::params_ns::dictionary::value_mismatch` is thrown.

            */
            params(int argc, const char* const* argv, const char* hdf5_path="/parameters");

            /// Access to argv[0] (returns empty string if unknown)
            std::string get_argv0() const;

            /// Access to ini file names (if any); returns empty string if out of range
            std::string get_ini_name(int n) const;

            /// Returns the number of ini file names
            int get_ini_name_count() const;


            /// Convenience method: returns the "origin name"
            /** @returns (parameter_file_name || restart_file name || program_name || "")

                @deprecated Use `alps::params_ns::origin_name(const params&)` instead,
                also available as `alps::origin_name(const params&)`.
             **/
            std::string get_origin_name() const ALPS_DEPRECATED;

            // FIXME: 1) Make these exception types derived from some `params::exception`;
            //        2) Define them ouside and typedef here.

            /// Exception type: the object was not restored from archive
            struct not_restored : public std::runtime_error {
                not_restored(const std::string& a_what)
                    : std::runtime_error(a_what) {}
            };

            /// Exception type: attempt to restore from 2 archives
            class archive_conflict : public std::runtime_error {
                std::vector<std::string> fnames_;
              public:
                archive_conflict(const std::string& a_what, const std::string& fname1, const std::string& fname2)
                    : std::runtime_error(a_what+"; name1='"+fname1+"' name2='"+fname2+"'"),
                      fnames_({fname1, fname2})
                { }

                const std::string& get_name(unsigned int i) const {
                    return fnames_[i % fnames_.size()];
                }

                const std::vector<std::string>& get_names() const {
                    return fnames_;
                }
            };

            /// Conveninece method: true if the object was restored from an archive
            bool is_restored() const { return !origins_.data()[origins_type::ARCHNAME].empty(); }

            /// Convenience method: returns the archive name the object has been restored from, or throws
            std::string get_archive_name() const;

            /// No-errors status
            bool ok() const { return err_status_.empty(); }

            /// True if there are missing or wrong-type parameters
            bool has_missing() const { return !ok(); }

            /// True if the parameter acquired its value by default
            bool defaulted(const std::string& name) const;

            /// True if the parameter is supplied via file or cmdline
            bool supplied(const std::string& name) const;

            /// True if there are parameters supplied but not defined; prints them out
            bool has_unused(std::ostream& out) const;

            /// True if there are parameters supplied but not defined in a subsection; prints them out
            /**
               @param out Stream to print the list of supplied, but unused parameters
               @param subsection The subsection to look into; empty
               string means "top level" (or "anonymous") subsection.
             */
            bool has_unused(std::ostream& out, const std::string& subsection) const;

            /// True if there are missing or wrong-type parameters; prints the message to that effect
            bool has_missing(std::ostream& out) const;

            /// True if user requested help
            bool help_requested() const;

            /// True if user requested help; print it to the supplied stream
            bool help_requested(std::ostream&) const;

            /// Print help to the given stream. @returns the stream
            std::ostream& print_help(std::ostream&) const;

            /// Returns true if the objects are identical
            bool operator==(const params& rhs) const;

            /// Check whether a parameter was ever defined
            /** That is, calling `define()` is unnecessary and will throw if type does not match. */
            bool defined(const std::string& name) const;

            /// Defines a parameter; returns false on error, and records the error in the object
            template<typename T>
            params& define(const std::string& name, const std::string& descr);

            /// Defines a parameter with a default; returns false on error, and records the error in the object
            template<typename T>
            params& define(const std::string& name, const T& defval, const std::string& descr);

            /// Defines a flag (boolean option with default of `false`)
            params& define(const std::string& name, const std::string& descr) {
                return define<bool>(name, false, descr);
            }

            /// Sets a description for the help message and introduces "--help" flag (if not already defined)
            params& description(const std::string& message);

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

            /// Collective (broadcasting) constructor from command line and parameter files.
            /** Reads and parses the command line on the root process,
                broadcasts to other processes. Tries to see if the
                file is an HDF5, in which case restores the object
                from the HDF5 file, ignoring the command line.

                @param comm : Communicator to use for broadcast
                @param root : Root process to broadcast from
                @param hdf5_path : path to HDF5 dataset containing the
                                 saved parameter object
                                 (NULL if this functionality is not needed)
            */
            params(int argc, const char* const* argv, const alps::mpi::communicator& comm, int root=0, const char* hdf5_path="/parameters")
                : dictionary(),
                  raw_kv_content_(),
                  td_map_(),
                  err_status_(),
                  origins_(),
                  help_header_()

            {
                initialize_(argc,argv,hdf5_path);
                broadcast(comm, root);
            }
#endif
        };

        /// Convenience function to obtain the "origin" filename associated with the parameters object
        /**
           The "origin" name can be used to generate, e.g., sensible output file names
           based on the parameter file names that passed to the program.

           * * If the parameters object is restored from an archive, the archive name is its origin.
           * * If the parameters object is constructed from INI file(s), the first INI file is its origin.
           * * If the parameters object is constructed from the command line without INI files,
             the origin is the executable name (if available) *stripped of its path*.
           * * Otherwise, the origin name is empty.

        */
        std::string origin_name(const params& p);

    } // params_ns::
    typedef params_ns::params params;

#ifdef ALPS_HAVE_MPI
    namespace mpi {

        inline void broadcast(const alps::mpi::communicator& comm, alps::params_ns::detail::td_type& td, int root) {
            broadcast(comm, td.typestr_, root);
            broadcast(comm, td.descr_, root);
            broadcast(comm, td.defnumber_, root);
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
