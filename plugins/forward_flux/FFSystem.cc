
#include "FFSystem.h"

// #include <pybind11/cast.h>
// #include <pybind11/stl_bind.h>

const uint64_t MAX_STEPS = 1000000;

namespace hoomd
    {

Scalar propensity(uint32_t pid, uint32_t mapped_pid, SnapshotParticleData<Scalar>& ref_snap, ParticleData* cur_pdata)
    {
    auto pid_pos = ArrayHandle<Scalar4>(cur_pdata->getPositions(),
        access_location::host, access_mode::read).data[pid];
    auto cur_pos = vec3<Scalar>(pid_pos.x, pid_pos.y, pid_pos.z);
    auto box = cur_pdata->getBox();
    vec3<Scalar> pos_diff = box.minImage(cur_pos - ref_snap.pos[pid]);
    return fast::sqrt(dot(pos_diff, pos_diff));
    }

FFSystem::FFSystem(std::shared_ptr<SystemDefinition> sysdef, uint64_t initial_tstep, uint32_t pid)
    : System(sysdef, initial_tstep), m_order_param(propensity), m_pid(pid)
    {
    }

std::optional<std::shared_ptr<SnapshotSystemData<Scalar>>> FFSystem::runFFTrial(const Scalar barrier, const std::shared_ptr<SnapshotSystemData<Scalar>> snapshot)
    {

    m_sysdef->initializeFromSnapshot(snapshot);
    auto old_time = m_cur_tstep;

    m_sysdef->getParticleData()->setFlags(determineFlags(m_cur_tstep));
    resetStats();
    
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // make sure we start off with a migration substep
        m_comm->forceMigrate();

        // communicate here
        m_comm->communicate(m_cur_tstep);
        }
#endif

    if (m_integrator)
        {
        m_integrator->prepRun(m_cur_tstep);
        }

    while (true)
        {
        for (auto& tuner : m_tuners)
            {
            if ((*tuner->getTrigger())(m_cur_tstep))
                tuner->update(m_cur_tstep);
            }

        if (m_update_group_dof_next_step)
            {
            updateGroupDOF();
            m_update_group_dof_next_step = false;
            }

        // look ahead to the next time step and see which analyzers and updaters will be executed
        // or together all of their requested PDataFlags to determine the flags to set for this time
        // step
        m_sysdef->getParticleData()->setFlags(determineFlags(m_cur_tstep + 1));

        // execute the integrator
        if (m_integrator)
            m_integrator->update(m_cur_tstep);

        auto op = computeOrderParameter();

        if (op < m_basin_barrier)
            {
            m_cur_tstep = old_time;
            return {};
            }
        else if (op >= barrier)
            {
            m_cur_tstep = old_time;
            auto new_snap = m_sysdef->takeSnapshot<Scalar>();
            return new_snap;
            }

        m_cur_tstep++;
        }
        
    }

std::vector<std::shared_ptr<SnapshotSystemData<Scalar>>> FFSystem::sampleBasinForwardFluxes(uint64_t nsteps)
    {
    std::vector<std::shared_ptr<SnapshotSystemData<Scalar>>> result = {};

    auto old_time = m_cur_tstep;

    m_sysdef->getParticleData()->setFlags(determineFlags(m_cur_tstep));
    resetStats();
    
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // make sure we start off with a migration substep
        m_comm->forceMigrate();

        // communicate here
        m_comm->communicate(m_cur_tstep);
        }
#endif

    if (m_integrator)
        {
        m_integrator->prepRun(m_cur_tstep);
        }

    auto last_op = computeOrderParameter();;

    for (int i = 0; i < nsteps; i++)
        {
        for (auto& tuner : m_tuners)
            {
            if ((*tuner->getTrigger())(m_cur_tstep))
                tuner->update(m_cur_tstep);
            }

        if (m_update_group_dof_next_step)
            {
            updateGroupDOF();
            m_update_group_dof_next_step = false;
            }

        // look ahead to the next time step and see which analyzers and updaters will be executed
        // or together all of their requested PDataFlags to determine the flags to set for this time
        // step
        m_sysdef->getParticleData()->setFlags(determineFlags(m_cur_tstep + 1));

        // execute the integrator
        if (m_integrator)
            m_integrator->update(m_cur_tstep);

        auto op = computeOrderParameter();

        if (op >= m_basin_barrier && last_op < m_basin_barrier)
            {
            auto new_snap = m_sysdef->takeSnapshot<Scalar>();
            result.push_back(new_snap);
            }

        last_op = op;

        m_cur_tstep++;
        }

    if (PyErr_CheckSignals() != 0)
        {
        throw pybind11::error_already_set();
        }

    m_cur_tstep = old_time;
    
    return result;
    }

void FFSystem::simpleRun(uint64_t nsteps)
    {

    // run the steps
    for (uint64_t count = 0; count < nsteps; count++)
        {
        for (auto& tuner : m_tuners)
            {
            if ((*tuner->getTrigger())(m_cur_tstep))
                tuner->update(m_cur_tstep);
            }

        if (m_update_group_dof_next_step)
            {
            updateGroupDOF();
            m_update_group_dof_next_step = false;
            }

        // look ahead to the next time step and see which analyzers and updaters will be executed
        // or together all of their requested PDataFlags to determine the flags to set for this time
        // step
        m_sysdef->getParticleData()->setFlags(determineFlags(m_cur_tstep + 1));

        // execute the integrator
        if (m_integrator)
            m_integrator->update(m_cur_tstep);

        m_cur_tstep++;

        }
    }

std::vector<Scalar> FFSystem::sampleBasin(uint64_t nsteps, uint64_t period)
    {
    auto n_writes = nsteps/period;
    std::vector<Scalar> result;
    result.reserve(n_writes);

    auto old_time = m_cur_tstep;
    // m_initial_time = m_clk.getTime();

    m_sysdef->getParticleData()->setFlags(determineFlags(m_cur_tstep));

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // make sure we start off with a migration substep
        m_comm->forceMigrate();

        // communicate here
        m_comm->communicate(m_cur_tstep);
        }
#endif

    if (m_update_group_dof_next_step)
        {
        updateGroupDOF();
        m_update_group_dof_next_step = false;
        }

    if (m_integrator)
        {
        m_integrator->prepRun(m_cur_tstep);
        }

    for (int i = 0; i < n_writes; i++) {
        simpleRun(period);
        result.push_back(computeOrderParameter());
    }

    // updateTPS();

    // if (PyErr_CheckSignals() != 0)
    //     {
    //     throw pybind11::error_already_set();
    //     }

    m_cur_tstep = old_time;

    return result;
    
    }


void FFSystem::setRefSnapshot()
    {
    m_ref_map = m_sysdef->getParticleData()->takeSnapshot(m_ref_snap);
    m_mapped_pid = m_ref_map[m_pid];
    }

void FFSystem::setRefSnapFromPython(std::shared_ptr<SnapshotSystemData<Scalar>> snapshot)
    {
    m_ref_snap = snapshot->particle_data;
    m_ref_map = snapshot->map;
    m_mapped_pid = m_ref_map[m_pid];
    }

void FFSystem::setBasinBarrier(const Scalar barrier)
    {
    m_basin_barrier = barrier;
    }

Scalar FFSystem::computeOrderParameter()
    {
    auto cur_pdata = m_sysdef->getParticleData();
    return m_order_param(m_pid, m_mapped_pid, m_ref_snap, cur_pdata.get());
    }

namespace detail
    {
void export_FFSystem(pybind11::module& m)
    {
    pybind11::class_<FFSystem, System, std::shared_ptr<FFSystem>>(m, "FFSystem")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, uint64_t, uint32_t>())

        .def("runFFTrial", &FFSystem::runFFTrial)
        .def("sampleBasin", &FFSystem::sampleBasin)
        .def("setBasinBarrier", &FFSystem::setBasinBarrier)
        .def("setRefSnapshot", &FFSystem::setRefSnapshot)
        .def("setRefSnapFromPython", &FFSystem::setRefSnapFromPython)
        .def("sampleBasinForwardFluxes", &FFSystem::sampleBasinForwardFluxes)

        .def("setIntegrator", &FFSystem::setIntegrator)
        .def("getIntegrator", &FFSystem::getIntegrator)

        .def("setAutotunerParams", &FFSystem::setAutotunerParams)
        .def("run", &FFSystem::run)

        .def("getLastTPS", &FFSystem::getLastTPS)
        .def("getCurrentTimeStep", &FFSystem::getCurrentTimeStep)
        .def("setPressureFlag", &FFSystem::setPressureFlag)
        .def("getPressureFlag", &FFSystem::getPressureFlag)
        .def_property_readonly("walltime", &FFSystem::getCurrentWalltime)
        .def_property_readonly("final_timestep", &FFSystem::getEndStep)
        .def_property_readonly("analyzers", &FFSystem::getAnalyzers)
        .def_property_readonly("updaters", &FFSystem::getUpdaters)
        .def_property_readonly("tuners", &FFSystem::getTuners)
        .def_property_readonly("computes", &FFSystem::getComputes)
#ifdef ENABLE_MPI
        .def("setCommunicator", &FFSystem::setCommunicator)
#endif
        ;
    }

    } // end namespace detail

    } // end namespace hoomd