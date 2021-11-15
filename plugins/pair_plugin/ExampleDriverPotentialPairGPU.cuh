#ifndef _EXAMPLE_DRIVER_POTENTIAL_PAIR_GPU_CUH_
#define _EXAMPLE_DRIVER_POTENTIAL_PAIR_GPU_CUH_

#include "EvaluatorPairExample.h"
#include "hoomd/md/PotentialPairGPU.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t __attribute__((visibility("default")))
gpu_compute_example_forces(const pair_args_t& pair_args,
                           const EvaluatorPairExample::param_type* d_params);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif // _EXAMPLE_DRIVER_POTENTIAL_PAIR_GPU_CUH_
