/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef PARAPACK_STAGING_H
#define PARAPACK_STAGING_H

#include <boost/filesystem/path.hpp>
#include <boost/tuple/tuple.hpp>
#include <queue>

namespace alps {
namespace parapack {

typedef
boost::tuple<int /* task_id */,int /* clone_id */,int /* group_id*/> suspended_queue_t;

void load_checkpoints(boost::filesystem::path const& file_chp,
		      boost::filesystem::path const& basedir,
		      std::queue<suspended_queue_t>& suspended_queue);

} // end namespace parapack
} // end namespace alps
#endif // PARAPACK_STAGING_H
