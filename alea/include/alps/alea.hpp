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
 * Types of accumulators
 * ---------------------
 * `alps::alea` defines a number of accumulators, which differ in the stored
 * statistical estimates and associated runtime and memory cost:
 *
 *   | Accumulator    | Runtime    | Memory     | mean | var | cov | tau |
 *   | -------------- | ---------- | ---------- | :--: | :-: | :-: | :-: |
 *   | `mean_acc`     | `N`        | `k`        |  X   |     |     |     |
 *   | `var_acc`      | `N`        | `k`        |  X   |  X  |     |     |
 *   | `cov_acc`      | `N`        | `k`        |  X   |  X  |  X  |     |
 *   | `autocorr_acc` | `a N`      | `k log(N)` |  X   |  X  |  X  |  X  |
 *   | `batch_acc`    | `a N`      | `k b`      |  X   |  X  |  X  | (X) |
 *
 * where in the complexity we defined the following terms:
 *
 *   - `N`: number of samples or calls to `operator<<`, i.e., final `count()`
 *   - `k`: components of the result vector, i.e., `size()`
 *   - `b`: number of batches, i.e., `num_batches()`
 *   - `a`: granularity factor (usually 2)
 *
 * and the following statistcal estimates:
 *
 *   - `mean`: sample mean
 *   - `count`: number of samples (whenever `mean` is available)
 *   - `var`: bias-corrected sample variance
 *   - `stderror`: standard error of the mean (whenever `var` is available)
 *   - `cov`: bias-corrected sample variance--covariance matrix
 *   - `tau`: integrated autocorrelation time
 *
 *
 * Accumulators and results
 * ------------------------
 * Most accumulators (`mean_acc`) have a matching result class (`mean_result`).
 * Accumulators and results differ conceptually and thus have almost orthogonal
 * functionality:
 *
 *  - accumulators support adding data to it, but no statistical estimates are
 *    available and no transformations can be performed on it.
 *
 *  - results do not support adding data, but one can perform transformations
 *    and reductions on it, as well as extracting mean, variance, etc.
 *
 * To obtain a result from an accumulator, the accumulators provide both a
 * `result()` and a `finalize()` method, where
 *
 *  1. the `result()` method creates an intermediate result, which leaves the
 *     accumulator untouched and thus must involve a copy of the data, while
 *
 *  2. the `finalize()` method invalidates the accumulator and thus allows to
 *     repurpose its data as the simulation result.  The `reset()` method then
 *     re-creates an empty accumulator with the same size.
 *
 * This can be represented by the following finite state machine:
 *
 *                     c'tor   _______________
 *                    ------->|               |
 *     result, <<        <<   |     empty     |
 *      +-------+       +-----|_______________|
 *      |       |       |            | |
 *      |     __V_______V____  reset | | reset  ________________
 *      |    |               |--->---+ +---<---|                |
 *      +----|  accumulating |                 |     invalid    |
 *           |_______________|---------------->|________________|
 *                                finalize
 *
 * @see alps::alea::mean_acc<T>::result(), alps::alea::mean_acc<T>::finalize()
 *
 *
 * Transformation and propagation of uncertainty
 * ---------------------------------------------
 * Transformations on results can be mediated using the `transform` method,
 * which takes a `transformer` instance.  Schematically:
 *
 *     OutResult transform(PropagationStrategy str, Transformer t, InResult in);
 *
 * Care has to be taken to correctly propagate the uncertainties. `alps::alea`
 * provides the following strategies, which differ in demand on the results,
 * complexity, and quality of bias correction:
 *
 *   | Strategy         | Prior | Bias   | Requires  | Invoc. |
 *   | -----------------|-------|--------|-----------|--------|
 *   | `no_prop`        | any   | -      | `mean()`  | 1      |
 *   | `linear_prop`    | Gauss | 1      | `var()`   | k      |
 *   | `sampling_prop`  | Gauss | 1/S    | `var()`   | S      |
 *   | `jackknife_prop` | any   | 1/b    | `batch()` | b      |
 *   | `bootstrap_prop` | any   | 1/S    | `batch()` | S      |
 *
 * Transformations must have one argument; functions of multiple random
 * variables (X,Y) can be realized by grouping the arguments together using
 * `alps::alea::join` and then applying the transform on the combined result.
 *
 * @see alps::alea::transform, alps::alea::join
 *
 *
 * Reduction and serialization
 * ---------------------------
 * All estimators support reduction (averaging over elements) through the
 * `reduce()` method, which takes the abstract `alps::alea::reducer` interface.
 * Depending on the implementation of the reducer, the reduction is performed
 * over different instances (threads, processes, etc.) using MPI, OpenMP, shell
 * etc.
 *
 * Similarly, all estimators support serialization (converting to permanent
 * format) though the `serialize()` method, which takes the abstract
 * `alps::alea::serializer` interface.
 *
 * @see alps::alea::reducer, alps::alea::serializer
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
#ifdef ALPS_HAVE_MPI
    #include <alps/alea/mpi.hpp>
#endif

// Variant types
#include <alps/alea/result.hpp>

// Transforms
#include <alps/alea/transform.hpp>
