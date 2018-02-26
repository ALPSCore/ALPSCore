/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_UTILITY_TEMPORARY_FILENAME_HPP
#define ALPS_UTILITY_TEMPORARY_FILENAME_HPP

#include <alps/config.hpp>
#include <string>
#include <boost/noncopyable.hpp>

namespace alps {
    namespace testing {
        /** @brief Resource manager: manages a file with unique name */
        class unique_file : private boost::noncopyable {
          public:
            enum action_type {
                REMOVE_AFTER, ///< Remove the file when destroying the object
                KEEP_AFTER, ///< Keep the file when destroying the object
                REMOVE_NOW, ///< Remove the file after constructing the object, remove when destructing too
                REMOVE_AND_DISOWN ///< Remove the file after constructing the object only 
            };

          private:
            std::string name_;
            action_type action_;
            
          public:
            /** @brief Generates a random file name with a given prefix
                @param prefix The file prefix
                @param action Whether to delete file in dtor
            */
            explicit unique_file(const std::string& prefix, action_type action=KEEP_AFTER);

            /// Returns temporary file name
            const std::string& name() const { return name_; }

            /// Closes and optionally deletes the file
            ~unique_file();
        };


        /// Convenience function: creates unique named file, returns its name
        inline std::string temporary_filename(const std::string& prefix) {
            unique_file uf(prefix,unique_file::KEEP_AFTER);
            return uf.name();
        }
            
            
    } // testing::
} // alps::
#endif // ALPS_UTILITY_TEMPORARY_FILENAME_HPP
