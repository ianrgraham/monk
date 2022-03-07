#ifndef _HERTZIAN_DRIVER_POTENTIAL_PAIR_GPU_CUH_
#define _HERTZIAN_DRIVER_POTENTIAL_PAIR_GPU_CUH_

#include "EvaluatorPairHertzian.h"
#include "hoomd/md/PotentialPairGPU.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t __attribute__((visibility("default")))
gpu_compute_hertzian_forces(const pair_args_t& pair_args,
                           const EvaluatorPairHertzian::param_type* d_params);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif // _HERTZIAN_DRIVER_POTENTIAL_PAIR_GPU_CUH_
