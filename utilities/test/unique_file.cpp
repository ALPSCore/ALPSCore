/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <fstream>
#include <alps/testing/unique_file.hpp>
#include <memory>
#include <map>

#include <gtest/gtest.h>

static const std::string Prefix="alps_temp_filename_test";

namespace at=alps::testing;

TEST(UniqueFile, RemoveAfter)
{
    std::string name; 
    {
        at::unique_file ufile(Prefix, at::unique_file::REMOVE_AFTER);
        name=ufile.name();
        std::ifstream fs(name.c_str(),std::ios_base::in|std::ios_base::out);
        EXPECT_TRUE(!!fs) << "Failed to open file '" << name << "'";
    }
    std::ifstream fs(name.c_str(),std::ios_base::in);
    EXPECT_FALSE(!!fs) << "File '" << name << "' should have been removed";
}

TEST(UniqueFile, RemoveNow)
{
    std::string name; 
    {
        at::unique_file ufile(Prefix, at::unique_file::REMOVE_NOW);
        name=ufile.name();
        {
            std::ifstream fs(name.c_str(),std::ios_base::in|std::ios_base::out);
            EXPECT_FALSE(!!fs) << "File '" << name << "' exists while it should not";
        }
        std::ifstream fs(name.c_str(),std::ios_base::out|std::ios_base::trunc);
        ASSERT_TRUE(!!fs) << "Failure to create file '" << name << "'";
    }
    std::ifstream fs(name.c_str(),std::ios_base::in);
    EXPECT_FALSE(!!fs) << "File '" << name << "' should have been removed";
}

TEST(UniqueFile, Disown)
{
    std::string name; 
    {
        at::unique_file ufile(Prefix, at::unique_file::REMOVE_AND_DISOWN);
        name=ufile.name();
        {
            std::ifstream fs(name.c_str(),std::ios_base::in|std::ios_base::out);
            EXPECT_FALSE(!!fs) << "File '" << name << "' exists while it should not";
        }
        std::ifstream fs(name.c_str(),std::ios_base::out|std::ios_base::trunc);
        ASSERT_TRUE(!!fs) << "Failure to create file '" << name << "'";
    }
    std::ifstream fs(name.c_str(),std::ios_base::in);
    EXPECT_TRUE(!!fs) << "File '" << name << "' should not have been removed";
    std::remove(name.c_str());
}

TEST(UniqueFile, KeepAfter)
{
    std::string name; 
    {
        at::unique_file ufile(Prefix, at::unique_file::KEEP_AFTER);
        name=ufile.name();
        std::ifstream fs(name.c_str(),std::ios_base::in|std::ios_base::out);
        EXPECT_TRUE(!!fs) << "Failed to open file '" << name << "'";
    }
    std::ifstream fs(name.c_str(),std::ios_base::in);
    EXPECT_TRUE(!!fs) << "File '" << name << "' should not have been removed";
    std::remove(name.c_str());
}

TEST(UniqueFile, KeepDefault)
{
    std::string name; 
    {
        at::unique_file ufile(Prefix);
        name=ufile.name();
        std::ifstream fs(name.c_str(),std::ios_base::in|std::ios_base::out);
        EXPECT_TRUE(!!fs) << "Failed to open file '" << name << "'";
    }
    std::ifstream fs(name.c_str(),std::ios_base::in);
    EXPECT_TRUE(!!fs) << "File '" << name << "' should not have been removed";
    std::remove(name.c_str());
}

TEST(TemporaryFile, CheckClash)
{
    const std::string prefix="alps_temp_filename_test";
    const unsigned int n_uniq_names=(1<<10); // how many uniq names to create

    typedef std::shared_ptr<at::unique_file> ufile_ptr;
    typedef std::map<std::string,ufile_ptr> map_type;
    typedef map_type::value_type val_type;
    typedef map_type::iterator iter_type;
    typedef std::pair<iter_type,bool> res_type;
    
    map_type nameset;
    for (unsigned int i=0; i<n_uniq_names; ++i) {
        ufile_ptr ufptr(new at::unique_file(prefix, at::unique_file::REMOVE_AFTER));
        std::string fname=ufptr->name();
        ASSERT_EQ(prefix, fname.substr(0,prefix.size()));
        
        res_type result=nameset.insert(val_type(fname,ufptr));
        ASSERT_TRUE(result.second) << "Clash for i=" << i << " name=" << fname;
    }
}
