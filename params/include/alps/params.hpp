/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_PARAMS_HPP_INCLUDED_8fd4d6abf4b6438cb3406e5a2d328d35
#define ALPS_PARAMS_HPP_INCLUDED_8fd4d6abf4b6438cb3406e5a2d328d35

#include "alps/config.hpp"
#include "alps/hdf5/archive.hpp"

#ifdef ALPS_HAVE_MPI
#include "alps/utilities/mpi.hpp"
#endif

#include "boost/program_options.hpp"
#include "boost/any.hpp"
#include "boost/tokenizer.hpp"

#include <map>
#include <vector>
#include <string>
#include <algorithm>

#include "./params/option_type.hpp"
#include "./params/option_description_type.hpp"
#include "./params/param_iterators.hpp"

namespace alps {
    namespace params_ns {

        class params {
          private:
            /// Type for the set of key names. Used in recording defaulted options
            // (FIXME: think of a better solution)
            typedef std::set<std::string> keys_set_type;

            /// True if there are no new define<>()-ed parameters since last parsing. Mutated by deferred parsing: certainly_parse()
            mutable bool is_valid_;
            /// Options (parameters). Mutated by deferred parsing: certainly_parse()
            mutable options_map_type optmap_;

            /// Set of options having their default value. Mutated by deferred parsing: certainly_parse()
            // FIXME: it is a quick hack. Should be merged with descr_map_ below.
            mutable keys_set_type defaulted_options_;

            /// Map (option names --> definition). Filled by define<T>() method.
            detail::description_map_type descr_map_;

            std::string helpmsg_;                 ///< Help message
            std::vector<std::string> argvec_;     ///< Command line arguments
            std::string file_content_;              ///< Content of INI file
            std::string infile_;                  ///< File name to read from (if not empty)
            std::string argv0_;                   ///< 0-th argument (program name)
            boost::optional<std::string> archname_; ///< Archive name (if restored from archive)

            /// Parses the parameter file, filling the option map, and using the provided options_description instance
            void certainly_parse(boost::program_options::options_description&, bool reassign=false) const;

            /// Parses the parameter file, filling the option map.
            void certainly_parse(bool reassign=false) const;

            /// Parses the parameters if not already parsed.
            void possibly_parse() const;

            /// Invalidates the option map
            void invalidate();

            /// Reads and pre-parses the INI file
            void preparse_ini();

            /// Initialization code common for all constructors
            void init();

            /// Initialization code used in constructors
            void init(unsigned int argc, const char* const* argv, const char* hdfpath);

            /// Function to check for validity/redefinition of an option (throws!)
            void check_validity(const std::string& optname) const;

          public:

            typedef options_map_type::iterator iterator;
            typedef options_map_type::const_iterator const_iterator;
            typedef options_map_type::value_type value_type;
            typedef options_map_type::mapped_type mapped_type;

            /// Iterator over names of "missing" options
            typedef detail::iterators::missing_params_iterator missing_params_iterator;

            // Some more convenience typedefs (exception types)
            /// Exception type: mismatched parameter types
            typedef option_type::type_mismatch type_mismatch;
            /// Exception type: attempt to use uninitialized option
            typedef option_type::uninitialized_value uninitialized_value;

            /// Exception type: attempt to redefine a parameter
            struct double_definition : public option_type::exception_base {
                double_definition(const std::string& a_name, const std::string& a_what)
                    : option_type::exception_base(a_name, a_what) {}
            };

            /// Exception type: attempt to define explicitly assigned parameter
            struct extra_definition : public option_type::exception_base {
                extra_definition(const std::string& a_name, const std::string& a_what)
                    : option_type::exception_base(a_name, a_what) {}
            };

            /// Exception type: incorrect parameter name
            struct invalid_name : public option_type::exception_base {
                invalid_name(const std::string& a_name, const std::string& a_what)
                    : option_type::exception_base(a_name, a_what) {}
            };

            /// Exception type: the object was not restored from archive
            struct not_restored : public std::runtime_error {
                not_restored(const std::string& a_what)
                    : std::runtime_error(a_what) {}
            };

            /** Default constructor */
            params();

            /** Constructor from HDF5 archive. */
            params(hdf5::archive ar, std::string const & path = "/parameters");

            /// Constructor from command line and a parameter file. The parsing is deferred.
            /** Tries to see if the file is an HDF5, in which case restores the object from the
                HDF5 file, ignoring the command line.

                @param hdfpath : path to HDF5 dataset containing the saved parameter object
                (NULL if this functionality is not needed)
            */
            params(unsigned int argc, const char* const* argv, const char* hdfpath = "/parameters");

#if defined(ALPS_HAVE_MPI)
            /// Collective constructor from command line and a parameter file. The parsing is deferred.
            /** Reads and parses the command line on the root process,
                broadcasts to other processes. Tries to see if the
                file is an HDF5, in which case restores the object
                from the HDF5 file, ignoring the command line.

                @param comm : Communicator to use for broadcast
                @param root : Root process to broadcast from
                @param hdfpath : path to HDF5 dataset containing the
                                 saved parameter object
                                 (NULL if this functionality is not needed)
            */
            params(unsigned int argc, const char* const* argv, const alps::mpi::communicator& comm,
                   int root=0, const char* hdfpath = "/parameters");
#endif

            /// Constructor from INI file (without command line)
            params(const std::string& inifile);

            /// Returns whether the parameters are restored from archive by the constructor
            bool is_restored() const;

            /// Returns the name of the archive the parameters were restarted from (or throw)
            std::string get_archive_name() const;

            /// @brief Convenience function: returns the "origin name"
            /// @Returns (parameter_file_name || restart_file name || program_name || "")
            std::string get_origin_name() const;

            /** Returns number of parameters (size of the map) */
            std::size_t size() const;

            /** Access a parameter: read-only */
            const mapped_type& operator[](const std::string& k) const;

            /** Access a parameter: possibly for assignment */
            mapped_type& operator[](const std::string& k);

            /** Returns iterator to the beginning of the option map */
            const_iterator begin() const;

            /** Iterator to the beyond-the-end of the option map */
            const_iterator end() const;

            /** Returns iterator over "missing" parameters */
            missing_params_iterator begin_missing() const;

            /** Returns iterator to beyond-the-end of "missing" parameters */
            missing_params_iterator end_missing() const;


            /// Check if a parameter exists (that is: attempt to read a compatible-typed value from it will not throw)
            bool exists(const std::string& name) const;

            /// Check if a parameter exists and is convertible to type T (that is: attempt to read T from it will not throw)
            template <typename T>
            bool exists(const std::string& name) const;

            /// @brief Check if parameter has default value (because it's not present in the command line).
            /// @details Returns false for unknown and missing parameters
            /// @remark Undefined (currently: false) for implicitly defined (by assignment) parameters
            // FIXME: what if it does not exist? what if it was explicitly assigned?
            bool defaulted(const std::string& name) const;

            /// Check if the parameter was ever `define()`-d or has a pre-assigned value (attempt to define() will throw)
            bool defined(const std::string& name) const;

            /// Save parameters to HDF5 archive
            void save(hdf5::archive &) const;

            /// Save parameters to HDF5 archive to a given path
            void save(hdf5::archive &, const std::string&) const;

            /// Load parameters from HDF5 archive (overwriting the object)
            void load(hdf5::archive &);

            /// Load parameters from HDF5 archive (overwriting the object) from a given path
            void load(hdf5::archive &, const std::string&);

#ifdef ALPS_HAVE_MPI
            /// Broadcast the parameters to all processes (FIXME: does not have a test yet)
            void broadcast(alps::mpi::communicator const &, int = 0);
#endif

            // -- now for the defining the options ---

            /// Set the help text for '--help' option
            params& description(const std::string& helpline);

            /// Define an option of type T with an optional value
            template <typename T>
            params& define(const std::string& optname, T defval, const std::string& descr);

            /// Define an option of type T with a required value
            template <typename T>
            params& define(const std::string& optname, const std::string& descr);

            /// Define a "trigger" command-line option (like "--help")
            params& define(const std::string& optname, const std::string& a_descr);

            /// Output the help message, if requested. @param ostrm: Output stream @returns true if help was indeed requested.
            bool help_requested(std::ostream& ostrm) const;

            /// Check if help requested. @returns true if help was requested.
            bool help_requested() const;

            /// Output the help message. @param ostrm: Output stream
            void print_help(std::ostream& ostrm) const;

            /// Output the list of missing parameters, if any. @param ostrm: Output stream @returns true if there are missing parameters
            bool has_missing(std::ostream& ostrm) const;

            /// Stream parameters
            friend std::ostream& operator<<(std::ostream& str, params const& x);

            /// Pick parameter by name and apply a generic functor f to it.
            /// For any allowed parameter type T, f must be callable as
            /// f(const std::string& name, boost::optional<T> const& val, boost::optional<T> const& defval, const std::string& descr)
            template <typename F>
            friend void apply(const params& opts, const std::string& optname, F const& f);

            /// Apply a generic functor f to each defined parameter
            /// For any allowed parameter type T, f must be callable as
            /// f(const std::string& name, boost::optional<T> const& val, boost::optional<T> const& defval, const std::string& descr)
            template <typename F>
            friend void foreach(const params& opts, F const& f);
        };

        // FIXME: we may consider provide template specializations for specific types? To hide templates inside *.cpp?
    } // params_ns::
} // alps::

// Inline member functions:
#include "./params/params_impl.hpp"

// Other implementation details
# include "./params/params_detail.hpp"

namespace alps {
    // Elevate relevant class to the alps NS:
    using params_ns::params;

} // alps::


#endif /* ALPS_PARAMS_HPP_INCLUDED_8fd4d6abf4b6438cb3406e5a2d328d35 */
