
#include "hoomd/System.h"
#include "hoomd/SnapshotSystemData.h"

// #include <types.h>
#include <optional>

#ifndef __FF_SYSTEM_H__
#define __FF_SYSTEM_H__

/*! \file FFSystem.h
    \brief Declares the Forward Flux System class and associated helper classes
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {

Scalar propensity(uint32_t, uint32_t, ParticleData&, ParticleData*);

std::vector<Scalar> sys_propensity(SnapshotParticleData<Scalar>&, std::map<unsigned int, unsigned int>&, ParticleData*);

/** In order to do this all rhobustly in MD, I think we need to constain this to
    methods where we can apply FIRE (or GD)
*/
class PYBIND11_EXPORT FFSystem: public System
    {
    public:
    //! Constructor
    FFSystem(std::shared_ptr<SystemDefinition> sysdef, uint64_t initial_tstep, uint32_t pid);

    //! Run the forward flux calculation segment, until the op evaluates
    std::pair<std::optional<std::shared_ptr<SnapshotSystemData<Scalar>>>, Scalar> runFFTrial(
        const Scalar barrier,
        const std::shared_ptr<SnapshotSystemData<Scalar>> snapshot,
        bool reset_tstep);

    std::vector<Scalar> sampleBasin(uint64_t nsteps, uint64_t period);

    std::vector<std::shared_ptr<SnapshotSystemData<Scalar>>> sampleBasinForwardFluxes(uint64_t nsteps);

    void setRefSnapshot();

    void setRefSnapFromPython(std::shared_ptr<SnapshotSystemData<Scalar>> snapshot);

    void setBasinBarrier(const Scalar barrier);

    void setPID(uint32_t pid);

    uint32_t getMappedPID();

    Scalar computeOrderParameter();

    private:

    //! Order parameter that is applied during FF calculation
    //  
    Scalar (*m_order_param)(uint32_t, uint32_t, SnapshotParticleData<Scalar>&, ParticleData*);
    uint32_t m_pid;
    uint32_t m_mapped_pid;
    std::optional<SnapshotParticleData<Scalar>> m_ref_snap;
    std::optional<std::map<unsigned int, unsigned int>> m_ref_map;
    Scalar m_basin_barrier;

    void simpleRun(uint64_t nsteps);
    
    };

namespace detail
    {

void export_FFSystem(pybind11::module& m);

    } // end namespace detail
    } // end namespace hoomd

#endif // __FF_SYSTEM_H__