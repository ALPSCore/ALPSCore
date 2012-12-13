/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                           Matthias Troyer <troyer@comp-phys.org>                *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef ALPS_NGS_HPP
#define ALPS_NGS_HPP

#include <alps/ngs/api.hpp>
#include <alps/ngs/cast.hpp>
#include <alps/ngs/sleep.hpp>
#include <alps/ngs/signal.hpp>
#include <alps/ngs/boost_mpi.hpp>
#include <alps/ngs/short_print.hpp>
#include <alps/ngs/thread_exceptions.hpp>
#include <alps/ngs/observablewrappers.hpp> // TODO: remove these:

#include <alps/ngs/alea.hpp>

// TODO: remove these:
#include <alps/ngs/scheduler/mcbase.hpp>
#include <alps/ngs/scheduler/mpimcsim.hpp>
#include <alps/ngs/scheduler/stop_callback.hpp>
#include <alps/ngs/scheduler/progress_callback.hpp>
//#include <alps/ngs/scheduler/mpiparallelsim.hpp>
//#include <alps/ngs/scheduler/multithreadedsim.hpp>

// TODO: remove these deprecated headers:
#include <alps/ngs/mcresult.hpp>
#include <alps/ngs/mcresults.hpp>
#include <alps/ngs/mcoptions.hpp>
#include <alps/ngs/mcobservable.hpp>
#include <alps/ngs/mcobservables.hpp> // TODO: rethink this!

#endif
