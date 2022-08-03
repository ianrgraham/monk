// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "EvaluatorPairExample.h"
#include "EvaluatorPairHertzian.h"
#include "EvaluatorPairFrictionLJ.h"
#include "hoomd/md/PotentialPair.h"
#include "FrictionPotentialPair.h"

#include <pybind11/pybind11.h>

#ifdef ENABLE_HIP
#include "hoomd/md/PotentialPairGPU.h"
// #include "ExampleDriverPotentialPairGPU.cuh"
// #include "HertzianDriverPotentialPairGPU.cuh"
#endif

namespace hoomd
    {
namespace md
    {

// specify the python module. Note that the name must explicitly match the PROJECT() name provided
// in CMakeLists (with an underscore in front)
PYBIND11_MODULE(_pair_plugin, m)
    {
    detail::export_PotentialPair<EvaluatorPairExample>(m, "PotentialPairExample");
    detail::export_PotentialPair<EvaluatorPairHertzian>(m, "PotentialPairHertzian");
    detail::export_FrictionPotentialPair<EvaluatorPairFrictionLJ>(m, "PotentialPairFrictionLJ");
#ifdef ENABLE_HIP
    detail::export_PotentialPairGPU<EvaluatorPairExample>(m, "PotentialPairExampleGPU");
    detail::export_PotentialPairGPU<EvaluatorPairHertzian>(m, "PotentialPairHertzianGPU");
    // TODO, write GPU implementation
    // detail::export_FrictionPotentialPairGPU<EvaluatorPairFrictionLJ>(m, "PotentialPairFrictionLJGPU");
#endif
    }

    } // end namespace md
    } // end namespace hoomd
