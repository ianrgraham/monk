
#include "FFSystem.h"

// #include <pybind11/cast.h>
// #include <pybind11/stl_bind.h>

namespace hoomd
    {

FFSystem::FFSystem(std::shared_ptr<SystemDefinition> sysdef, uint64_t initial_tstep)
    : System(sysdef, initial_tstep) {
}

void FFSystem::runFFTrial(const double barrier, const SystemDefinition& sysdef) {

    m_sysdef = std::make_shared<SystemDefinition>(sysdef);

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

    auto initial_state = m_sysdef->getParticleData();
    

}

void FFSystem::sampleBasin() {

}

namespace detail
    {
void export_FFSystem(pybind11::module& m)
    {
    pybind11::class_<FFSystem, std::shared_ptr<FFSystem>>(m, "FFSystem")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, uint64_t>())

        .def("runFFTrial", &FFSystem::runFFTrial)

        .def("setIntegrator", &FFSystem::setIntegrator)
        .def("getIntegrator", &FFSystem::getIntegrator)

        .def("setAutotunerParams", &FFSystem::setAutotunerParams)
        .def("enableProfiler", &FFSystem::enableProfiler)
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