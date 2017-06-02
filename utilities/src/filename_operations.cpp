/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


#include <alps/utilities/remove_extensions.hpp>
#include <alps/utilities/get_basename.hpp>
#include <alps/utilities/get_dirname.hpp>
#include <boost/filesystem.hpp>

namespace alps {
    std::string remove_extensions(const std::string& filename)
    {
        using std::string;
        typedef std::string::size_type size_type;
        size_type basedir_len=filename.rfind('/');
        if (basedir_len==string::npos)
            basedir_len=0;
        else if (basedir_len==filename.size()) // special case: "dir/"
            return filename;
        else
            ++basedir_len;

        // special cases: "dir/." and "dir/.."
        if (filename.compare(basedir_len, string::npos, ".")==0 ||
            filename.compare(basedir_len, string::npos, "..")==0)
            return filename;

        // special case: "dir/...ext" == "dir/.."+".ext" ("ext" may be "")
        if (filename.compare(basedir_len, 3, "...")==0)
            return filename.substr(0,basedir_len+2);

        // special case: "dir/..ext" == "dir/." + ".ext"
        if (filename.compare(basedir_len, 2, "..")==0)
            return filename.substr(0,basedir_len+1);
            
        size_type first_dot_pos=filename.find('.', basedir_len);
        return filename.substr(0,first_dot_pos); // works also if file has no extensions
    }

    std::string get_basename(const std::string& filename)
    {
        using std::string;
        typedef std::string::size_type size_type;
            
        size_type basedir_len=filename.rfind('/');
        if (basedir_len==string::npos) return filename;
        if (basedir_len+1==filename.size()) {
            return basedir_len==0 ? "/" : "."; // special case of "/"
        }
        return filename.substr(basedir_len+1);
    }

    std::string get_dirname(const std::string& filename)
    {
        using std::string;
        typedef std::string::size_type size_type;
            
        size_type basedir_len=filename.rfind('/');
        if (basedir_len==string::npos) return "";
        if (basedir_len==0 && filename.size()!=1) return "/"; // special case of "/file_in_root"

        return filename.substr(0,basedir_len);
    }
} // alps::
