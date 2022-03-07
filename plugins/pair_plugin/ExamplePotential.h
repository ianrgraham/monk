
#ifndef __EXAMPLE_PAIR_POTENTIAL_H__
#define __EXAMPLE_PAIR_POTENTIAL_H__

#include "EvaluatorPairExample.h"
#include "EvaluatorPairHertzian.h"
#include "hoomd/md/PotentialPair.h"

// #define ENABLE_HIP // to test

#ifdef ENABLE_HIP
#include "ExampleDriverPotentialPairGPU.cuh"
#include "HertzianDriverPotentialPairGPU.cuh"
#include "hoomd/md/PotentialPairGPU.h"
#endif

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {
//! Pair potential force compute for example forces
typedef PotentialPair<EvaluatorPairExample> PotentialPairExample;
typedef PotentialPair<EvaluatorPairHertzian> PotentialPairHertzian;

#ifdef ENABLE_HIP
//! Pair potential force compute for example forces on the GPU
typedef PotentialPairGPU<EvaluatorPairExample, kernel::gpu_compute_example_forces>
    PotentialPairExampleGPU;
typedef PotentialPairGPU<EvaluatorPairHertzian, kernel::gpu_compute_hertzian_forces>
    PotentialPairHertzianGPU;
#endif

    } // end namespace md
    } // end namespace hoomd

#endif // _EXAMPLE_PAIR_POTENTIAL_H__
