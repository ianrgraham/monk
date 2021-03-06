
#include "FFSystem.h"

const uint64_t MAX_STEPS = 1000000;

namespace hoomd
    {

std::optional<int> optionalInt(int x, bool b) {
    if (b) {
        return x;
    }
    else {
        return {};
    }
}

Scalar propensity(uint32_t pid, uint32_t mapped_pid, SnapshotParticleData<Scalar>& ref_snap, ParticleData* cur_pdata)
    {
    // rtags are a map from tags to indices
    // tags are a map from indices to tags
    // mapped_pid is the index of the particle in the current snapshot associated with the tag pid

    // so scratch what I said before. It appear that the SnapshotParticleData object sorts all of the data by tag
    // so there is no need to do a lookup by index.

    // so we get the index in the current frame of the pid with the rtags array
    auto idx = ArrayHandle<unsigned int>(cur_pdata->getRTags(),
        access_location::host, access_mode::read).data[pid];
    auto local_pos = ArrayHandle<Scalar4>(cur_pdata->getPositions(),
        access_location::host, access_mode::read).data[idx];
    auto cur_pos = vec3<Scalar>(local_pos.x, local_pos.y, local_pos.z);
    auto box = cur_pdata->getBox();

    // ref_snap is sorted by tags! just use the pid tag to get the position
    vec3<Scalar> pos_diff = box.minImage(cur_pos - ref_snap.pos[pid]);
    return fast::sqrt(dot(pos_diff, pos_diff));
    }

std::vector<Scalar> sys_propensity(SnapshotParticleData<Scalar>& ref_snap, std::map<unsigned int, unsigned int>& ref_map, ParticleData* cur_pdata)
    {
    auto box = cur_pdata->getBox();
    unsigned n_tot_particles = cur_pdata->getN();

    std::vector<Scalar> output{};
    output.reserve(n_tot_particles);
    
    // get the rtags array
    auto rtags = ArrayHandle<unsigned int>(cur_pdata->getRTags(),
        access_location::host, access_mode::read);

    // get the positions array
    auto pos = ArrayHandle<Scalar4>(cur_pdata->getPositions(),
        access_location::host, access_mode::read);

    for (unsigned int i = 0; i < n_tot_particles; i++)
        {
        // grab the index for tag 'i' in the current frame
        auto idx = rtags.data[i];
        auto local_pos = pos.data[idx];
        auto cur_pos = vec3<Scalar>(local_pos.x, local_pos.y, local_pos.z);
        // grab the index for tag 'i' in the reference snapshot, use i since ref_snap is already sorted by tag
        auto ref_pos = ref_snap.pos[i];
        vec3<Scalar> pos_diff = box.minImage(cur_pos - ref_pos);
        output.push_back(fast::sqrt(dot(pos_diff, pos_diff)));
        }
    return output;
    }

FFSystem::FFSystem(std::shared_ptr<SystemDefinition> sysdef, uint64_t initial_tstep, uint32_t pid)
    : System(sysdef, initial_tstep), m_order_param(propensity), m_sys_order_param(sys_propensity), m_pid(pid)
    {
    }

std::pair<std::optional<std::shared_ptr<SnapshotSystemData<Scalar>>>, Scalar> FFSystem::runFFTrial(const Scalar barrier, const std::shared_ptr<SnapshotSystemData<Scalar>> snapshot, bool reset_tstep)
    {

    m_sysdef->initializeFromSnapshot(snapshot);
    updateGroupDOFOnNextStep();
    auto old_time = m_cur_tstep;

    m_sysdef->getParticleData()->setFlags(determineFlags(m_cur_tstep));
    resetStats();

    auto op = computeOrderParameter();
    auto max_op = op;
    // std::cout << "Starting op: " << op << std::endl; 
    
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

    auto idx = 0;
    while (true)
        {
        // std::cout << "Loop idx: " << idx << ". Op: " << computeOrderParameter() << std::endl; 
        if (op < m_basin_barrier)
            {
            if (reset_tstep)
                {
                m_cur_tstep = old_time;
                }
            // std::cout << "  Failed @ idx" << idx << std::endl;
            // std::cout << "    Ops: " << op << "<" << m_basin_barrier << std::endl;
            return {{}, max_op};
            }
        else if (op >= barrier)
            {
            if (reset_tstep)
                {
                m_cur_tstep = old_time;
                }
            auto new_snap = m_sysdef->takeSnapshot<Scalar>();
            // std::cout << "  Success @ idx" << idx << std::endl;
            // std::cout << "    Ops: " << op << ">=" << barrier << std::endl;
            return {new_snap, max_op};
            }

        for (auto& tuner : m_tuners)
            {
            if ((*tuner->getTrigger())(m_cur_tstep))
                tuner->update(m_cur_tstep);
            }

        for (auto& updater : m_updaters)
            {
            if ((*updater->getTrigger())(m_cur_tstep))
                {
                updater->update(m_cur_tstep);
                m_update_group_dof_next_step |= updater->mayChangeDegreesOfFreedom(m_cur_tstep);
                }
            }

        // std::cout << "Update idx: " << idx << ". Op: " << computeOrderParameter() << std::endl; 

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
            m_integrator->update(rand());

        // std::cout << "Integration idx: " << idx << ". Op: " << computeOrderParameter() << std::endl; 

        op = computeOrderParameter();
        if (op > max_op) {
            max_op = op;
        }

        m_cur_tstep++;
        idx++;

        // execute analyzers after incrementing the step counter
        for (auto& analyzer : m_analyzers)
            {
            if ((*analyzer->getTrigger())(m_cur_tstep))
                analyzer->analyze(m_cur_tstep);
            }

        // propagate Python exceptions related to signals
        if (PyErr_CheckSignals() != 0)
            {
            throw pybind11::error_already_set();
            }

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

    auto last_op = computeOrderParameter();

    for (int i = 0; i < nsteps; i++)
        {
        for (auto& tuner : m_tuners)
            {
            if ((*tuner->getTrigger())(m_cur_tstep))
                tuner->update(m_cur_tstep);
            }

        for (auto& updater : m_updaters)
            {
            if ((*updater->getTrigger())(m_cur_tstep))
                {
                updater->update(m_cur_tstep);
                m_update_group_dof_next_step |= updater->mayChangeDegreesOfFreedom(m_cur_tstep);
                }
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

    // m_cur_tstep = old_time;
    
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

        // execute updaters
        for (auto& updater : m_updaters)
            {
            if ((*updater->getTrigger())(m_cur_tstep))
                {
                updater->update(m_cur_tstep);
                m_update_group_dof_next_step |= updater->mayChangeDegreesOfFreedom(m_cur_tstep);
                }
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

        // execute analyzers after incrementing the step counter
        for (auto& analyzer : m_analyzers)
            {
            if ((*analyzer->getTrigger())(m_cur_tstep))
                analyzer->analyze(m_cur_tstep);
            }

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

    if (PyErr_CheckSignals() != 0)
        {
        throw pybind11::error_already_set();
        }

    // m_cur_tstep = old_time;

    return result;
    
    }

std::vector<std::vector<Scalar>> FFSystem::sampleAllBasins(uint64_t nsteps, uint64_t period)
    {
    auto n_writes = nsteps/period;
    std::vector<std::vector<Scalar>> result;
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
        result.push_back(computeSysOrderParameter());
    }

    // updateTPS();

    if (PyErr_CheckSignals() != 0)
        {
        throw pybind11::error_already_set();
        }

    // m_cur_tstep = old_time;

    return result;
    
    }

void FFSystem::setRefSnapshot()
    {
    m_ref_map = m_sysdef->getParticleData()->takeSnapshot(*m_ref_snap);
    m_mapped_pid = m_ref_map->at(m_pid);
    }

uint32_t FFSystem::getMappedPID()
    {
    return m_mapped_pid;
    }

std::optional<std::map<unsigned int, unsigned int>> FFSystem::getRefMap()
    {
    return m_ref_map;
    }

void FFSystem::setRefSnapFromPython(std::shared_ptr<SnapshotSystemData<Scalar>> snapshot)
    {
    m_ref_snap = snapshot->particle_data;
    m_ref_map = snapshot->map;
    m_mapped_pid = m_ref_map->at(m_pid);
    }

void FFSystem::setBasinBarrier(const Scalar barrier)
    {
    m_basin_barrier = barrier;
    }

Scalar FFSystem::computeOrderParameter()
    {
    auto cur_pdata = m_sysdef->getParticleData();
    return m_order_param(m_pid, m_mapped_pid, *m_ref_snap, cur_pdata.get());
    }

std::vector<Scalar> FFSystem::computeSysOrderParameter()
    {
    auto cur_pdata = m_sysdef->getParticleData();
    return m_sys_order_param(m_ref_snap.value(), m_ref_map.value(), cur_pdata.get());
    }

void FFSystem::setPID(uint32_t pid)
    {
    m_pid = pid;
    if (m_ref_map) {
        m_mapped_pid = m_ref_map->at(m_pid);
    }
    }

namespace detail
    {
void export_FFSystem(pybind11::module& m)
    {
    m.def("optionalInt", &optionalInt);
    pybind11::class_<FFSystem, System, std::shared_ptr<FFSystem>>(m, "FFSystem")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, uint64_t, uint32_t>())

        .def("runFFTrial", &FFSystem::runFFTrial)
        .def("sampleBasin", &FFSystem::sampleBasin)
        .def("sampleAllBasins", &FFSystem::sampleAllBasins)
        .def("setBasinBarrier", &FFSystem::setBasinBarrier)
        .def("setRefSnapshot", &FFSystem::setRefSnapshot)
        .def("setRefSnapFromPython", &FFSystem::setRefSnapFromPython)
        .def("sampleBasinForwardFluxes", &FFSystem::sampleBasinForwardFluxes)
        .def("computeOrderParameter", &FFSystem::computeOrderParameter)
        .def("setPID", &FFSystem::setPID)
        .def("getMappedPID", &FFSystem::getMappedPID)
        .def("getRefMap", &FFSystem::getRefMap)
        ;
    }

    } // end namespace detail

    } // end namespace hoomd