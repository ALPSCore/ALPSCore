/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

/**
 * @namespace alps::alea
 * @brief Set of accumulators and statistical pre-/post-processing operations.
 *
 * Overview
 * --------
 * TODO
 *
 * Accumulators and results
 * ------------------------
 * Most accumulators (`mean_acc`) have a matching result class (`mean_result`).
 * To obtain a result from an accumulator, the accumulators provide both a
 * `result()` and a `finalize()` method, where
 *
 *  1. the `result()` method creates an intermediate result, which leaves the
 *     accumulator untouched and thus must involve a copy of the data, while
 *
 *  2. the `finalize()` method invalidates the accumulator and thus allows to
 *     repurpose its data as the simulation result.  The reset method then
 *     re-creates an empty accumulator with the same size.
 *
 * This can be represented by the following finite state machine:
 *
 *                     c'tor   _______________      _______________
 *                    ------->|               |    |               |  default
 *     result, <<        <<   |     empty     |    | uninitialized |   c'tor
 *      +-------+       +-----|_______________|    |_______________|<<-------
 *      |       |       |            | |
 *      |     __V_______V____  reset | | reset  ________________
 *      |    |               |--->---+ +---<---|                |
 *      +----|  accumulating |                 |     invalid    |
 *           |_______________|---------------->|________________|
 *                                finalize
 */

// Base
#include <alps/alea/core.hpp>

// Accumulator types
#include <alps/alea/mean.hpp>
#include <alps/alea/variance.hpp>
#include <alps/alea/covariance.hpp>
#include <alps/alea/autocorr.hpp>
#include <alps/alea/batch.hpp>

// Plugins
#include <alps/alea/hdf5.hpp>
#include <alps/alea/mpi.hpp>

// Variant types
#include <alps/alea/result.hpp>

