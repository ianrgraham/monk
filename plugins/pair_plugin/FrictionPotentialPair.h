// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __FRICTION_POTENTIAL_PAIR_H__
#define __FRICTION_POTENTIAL_PAIR_H__

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include "hoomd/md/NeighborList.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/GSDShapeSpecWriter.h"

#include "hoomd/ManagedArray.h"
#include "hoomd/VectorMath.h"

/*! \file FrictionPotentialPair.h
    \brief Defines the template class for frictional pair potentials
    \details The heart of the code that computes frictional pair potentials is in this file.
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Template class for computing pair potentials
/*! <b>Overview:</b>
    FrictionPotentialPair computes standard pair potentials (and forces) between all particle pairs in
   the simulation. It employs the use of a neighbor list to limit the number of computations done to
   only those particles with the cutoff radius of each other. The computation of the actual V(r) is
   not performed directly by this class, but by an friction_evaluator class (e.g. EvaluatorPairLJ)
   which is passed in as a template parameter so the computations are performed as efficiently as
   possible.

    FrictionPotentialPair handles most of the gory internal details common to all standard pair
   potentials.
     - A cutoff radius to be specified per particle type pair
     - The energy can be globally shifted to 0 at the cutoff
     - Per type pair parameters are stored and a set method is provided
     - And all the details about looping through the particles, computing dr, computing the virial,
   etc. are handled

    \note XPLOR switching is not supported

    <b>Implementation details</b>

    rcutsq and the params are stored per particle type pair. It wastes a little bit of space, but
   benchmarks show that storing the symmetric type pairs and indexing with Index2D is faster than
   not storing redundant pairs and indexing with Index2DUpperTriangular. All of these values are
   stored in GlobalArray for easy access on the GPU by a derived class. The type of the parameters
   is defined by \a param_type in the potential friction_evaluator class passed in. See the appropriate
   documentation for the friction_evaluator for the definition of each element of the parameters.
*/

// template<class Real> struct AxisAngle
//     {

//     AxisAngle::AxisAngle(const Real& _angle, const vec3<Real>& _axis) : angle(_angle), axis(_axis) { }

//     DEVICE static AxisAngle fromQuat(const quat<Real>& q, const vec3<Real>& axis, const Real& theta)
//         {
        
//         auto theta = fast::acos(q.s) * Real(2.0);
//         AxisAngle<Real> a(theta, q.v/fast::sin(theta/Real(2.0)));
//         return a;
//         }
    
//     Real angle;         //!< scalar component
//     vec3<Real> axis;    //!< vector component
//     };

struct FrictionResult
    {
    bool eval;
    bool contact;
    };

struct pair_hash
    {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const
        {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
        }
    };

template<class friction_evaluator> class FrictionPotentialPair : public ForceCompute
    {
    public:
    //! Param type from friction_evaluator
    typedef typename friction_evaluator::param_type param_type;

    //! Shape param type from friction_evaluator
    typedef typename friction_evaluator::shape_type shape_type;

    //! Construct the pair potential
    FrictionPotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<NeighborList> nlist);
    //! Destructor
    virtual ~FrictionPotentialPair();

    //! Set the pair parameters for a single type pair
    virtual void setParams(unsigned int typ1, unsigned int typ2, const param_type& param);

    virtual void setParamsPython(pybind11::tuple typ, pybind11::object params);

    /// Get params for a single type pair using a tuple of strings
    virtual pybind11::object getParamsPython(pybind11::tuple typ);

    //! Set the rcut for a single type pair
    virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);

    /// Get the r_cut for a single type pair
    Scalar getRCut(pybind11::tuple types);

    /// Set the rcut for a single type pair using a tuple of strings
    virtual void setRCutPython(pybind11::tuple types, Scalar r_cut);

    //! Method that is called whenever the GSD file is written if connected to a GSD file.
    int slotWriteGSDShapeSpec(gsd_handle&) const;

    /// Validate that types are within Ntypes
    virtual void validateTypes(unsigned int typ1, unsigned int typ2, std::string action);
    //! Method that is called to connect to the gsd write state signal
    void connectGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer);

    //! Set the shape parameters for a single type
    virtual void setShape(unsigned int typ, const shape_type& shape_param);

    virtual pybind11::object getShapePython(std::string typ);

    //! Set the shape parameters for a single type through Python
    virtual void setShapePython(std::string typ, const pybind11::object shape_param);

    std::vector<std::string> getTypeShapeMapping() const
        {
        std::vector<std::string> type_shape_mapping(m_pdata->getNTypes());
        for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
            {
            friction_evaluator eval(Scalar(0.0), Scalar(0.0), this->m_params[m_typpair_idx(i, i)]);
            type_shape_mapping[i] = eval.getShapeSpec();
            }
        return type_shape_mapping;
        }

    //! Shifting modes that can be applied to the energy
    enum energyShiftMode
        {
        no_shift = 0,
        shift,
        };

    //! Set the mode to use for shifting the energy
    void setShiftMode(energyShiftMode mode)
        {
        m_shift_mode = mode;
        }

    void setShiftModePython(std::string mode)
        {
        if (mode == "none")
            {
            m_shift_mode = no_shift;
            }
        else if (mode == "shift")
            {
            m_shift_mode = shift;
            }
        else
            {
            throw std::runtime_error("Invalid energy shift mode.");
            }
        }

    /// Get the mod eused for the energy shifting
    std::string getShiftMode()
        {
        switch (m_shift_mode)
            {
        case no_shift:
            return "none";
        case shift:
            return "shift";
        default:
            return "";
            }
        }

    void clearDynamicState() {
        m_dynamic_state_flag = false;
        m_xi.clear();
        m_psi.clear();
        m_pair_idx.clear();
    }

    virtual void notifyDetach()
        {
        if (m_attached)
            {
            m_nlist->removeRCutMatrix(m_r_cut_nlist);
            }
        if (!m_persist_state_on_detach) {
            clearDynamicState();
        }
        m_attached = false;
        }

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    virtual CommFlags getRequestedCommFlags(uint64_t timestep);
#endif

    //! Returns true because we compute the torque
    virtual bool isAnisotropic()
        {
        return true;
        }

    protected:
    std::shared_ptr<NeighborList> m_nlist; //!< The neighborlist to use for the computation
    energyShiftMode m_shift_mode; //!< Store the mode with which to handle the energy shift at r_cut
    Index2D m_typpair_idx;        //!< Helper class for indexing per type pair arrays
    GlobalArray<Scalar> m_rcutsq; //!< Cutoff radius squared per type pair
    std::vector<param_type, hoomd::detail::managed_allocator<param_type>>
        m_params; //!< Pair parameters per type pair
    std::vector<shape_type, hoomd::detail::managed_allocator<shape_type>>
        m_shape_params; //!< Shape paramters per type

    /// Track whether we have attached to the Simulation object
    bool m_attached = true;

    // THE FOUR VARIABLES BELOW ARE STATEFUL! WE NEED TO ENSURE THEY HAVE SENSIBLE VALUES AT ALL TIMES!
    // Be careful to engage/disengage the dynamic state flag when the force compute is added/removed from the system.

    bool m_dynamic_state_flag = false;
    bool m_persist_state_on_detach = false;

    // Dynamically track quantities relevant to contact friction
    // Angular momentum quaternion needs to be converted to real space frame vector for these computations
    std::vector<Scalar3, hoomd::detail::managed_allocator<Scalar3>> m_xi; //!< transverse surface velocity integrated
    std::vector<Scalar3, hoomd::detail::managed_allocator<Scalar3>> m_psi; //!< rotational surface velocity integrated

    /// Maps pairs of particles to indices in the surface velocities
    /// This will be necessary to resort the arrays after neighborlist updates
    std::unordered_map<std::pair<unsigned int, unsigned int>, unsigned int, pair_hash> m_pair_idx;

    // Might replace the above variable with a list 
    

    /// r_cut (not squared) given to the neighbor list
    std::shared_ptr<GlobalArray<Scalar>> m_r_cut_nlist;

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

template<class friction_evaluator>
void FrictionPotentialPair<friction_evaluator>::connectGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer)
    {
    typedef hoomd::detail::SharedSignalSlot<int(gsd_handle&)> SlotType;
    auto func = std::bind(&FrictionPotentialPair<friction_evaluator>::slotWriteGSDShapeSpec,
                          this,
                          std::placeholders::_1);
    std::shared_ptr<hoomd::detail::SignalSlot> pslot(new SlotType(writer->getWriteSignal(), func));
    addSlot(pslot);
    }

template<class friction_evaluator>
int FrictionPotentialPair<friction_evaluator>::slotWriteGSDShapeSpec(gsd_handle& handle) const
    {
    hoomd::detail::GSDShapeSpecWriter shapespec(m_exec_conf);
    m_exec_conf->msg->notice(10) << "FrictionPotentialPair writing to GSD File to name: "
                                 << shapespec.getName() << std::endl;
    int retval = shapespec.write(handle, this->getTypeShapeMapping());
    return retval;
    }

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
*/
template<class friction_evaluator>
FrictionPotentialPair<friction_evaluator>::FrictionPotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                                                        std::shared_ptr<NeighborList> nlist)
    : ForceCompute(sysdef), m_nlist(nlist), m_shift_mode(no_shift),
      m_typpair_idx(m_pdata->getNTypes())
    {
    m_exec_conf->msg->notice(5) << "Constructing FrictionPotentialPair<" << friction_evaluator::getName()
                                << ">" << std::endl;
    assert(m_pdata);
    assert(m_nlist);

    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);

    // initialize empty containers for xi, psi, and pair_idx
    std::vector<Scalar3, hoomd::detail::managed_allocator<Scalar3>> xi;
    std::vector<Scalar3, hoomd::detail::managed_allocator<Scalar3>> psi;
    std::unordered_map<std::pair<unsigned int, unsigned int>, unsigned int, pair_hash> pair_idx;
    auto num_nlist_elements = 0;
    m_xi.reserve(num_nlist_elements);
    m_psi.reserve(num_nlist_elements);
    m_pair_idx.reserve(m_pair_idx.max_load_factor()*num_nlist_elements);

    m_xi.swap(xi);
    m_psi.swap(psi);
    m_pair_idx.swap(pair_idx);

    GlobalArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_rcutsq.swap(rcutsq);
    GlobalArray<Scalar> ronsq(m_typpair_idx.getNumElements(), m_exec_conf);
    std::vector<param_type, hoomd::detail::managed_allocator<param_type>> params(
        static_cast<size_t>(m_typpair_idx.getNumElements()),
        param_type(),
        hoomd::detail::managed_allocator<param_type>(m_exec_conf->isCUDAEnabled()));
    m_params.swap(params);

    std::vector<shape_type, hoomd::detail::managed_allocator<shape_type>> shape_params(
        static_cast<size_t>(m_pdata->getNTypes()),
        shape_type(),
        hoomd::detail::managed_allocator<shape_type>(m_exec_conf->isCUDAEnabled()));
    m_shape_params.swap(shape_params);

    m_r_cut_nlist
        = std::make_shared<GlobalArray<Scalar>>(m_typpair_idx.getNumElements(), m_exec_conf);
    nlist->addRCutMatrix(m_r_cut_nlist);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled())
        {
        cudaMemAdvise(m_params.data(),
                      m_params.size() * sizeof(param_type),
                      cudaMemAdviseSetReadMostly,
                      0);
        cudaMemAdvise(m_shape_params.data(),
                      m_shape_params.size() * sizeof(shape_type),
                      cudaMemAdviseSetReadMostly,
                      0);

        // prefetch
        auto& gpu_map = m_exec_conf->getGPUIds();

        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            // prefetch data on all GPUs
            cudaMemPrefetchAsync(m_params.data(),
                                 sizeof(param_type) * m_params.size(),
                                 gpu_map[idev]);
            cudaMemPrefetchAsync(m_shape_params.data(),
                                 sizeof(shape_type) * m_shape_params.size(),
                                 gpu_map[idev]);
            }

        if (m_exec_conf->allConcurrentManagedAccess())
            {
            cudaMemAdvise(m_rcutsq.get(),
                          m_rcutsq.getNumElements() * sizeof(Scalar),
                          cudaMemAdviseSetReadMostly,
                          0);

            for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
                {
                // prefetch data on all GPUs
                cudaMemPrefetchAsync(m_rcutsq.get(),
                                     sizeof(Scalar) * m_rcutsq.getNumElements(),
                                     gpu_map[idev]);
                }
            }
        }
#endif
    }

template<class friction_evaluator> FrictionPotentialPair<friction_evaluator>::~FrictionPotentialPair()
    {
    m_exec_conf->msg->notice(5) << "Destroying FrictionPotentialPair<" << friction_evaluator::getName()
                                << ">" << std::endl;

    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param param Parameter to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
template<class friction_evaluator>
void FrictionPotentialPair<friction_evaluator>::setParams(unsigned int typ1,
                                                    unsigned int typ2,
                                                    const param_type& param)
    {
    validateTypes(typ1, typ2, "setting params");
    m_params[m_typpair_idx(typ1, typ2)] = param;
    m_params[m_typpair_idx(typ2, typ1)] = param;
    }

template<class friction_evaluator>
void FrictionPotentialPair<friction_evaluator>::setParamsPython(pybind11::tuple typ,
                                                          pybind11::object params)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    setParams(typ1, typ2, param_type(params, m_exec_conf->isCUDAEnabled()));
    }

template<class friction_evaluator>
pybind11::object FrictionPotentialPair<friction_evaluator>::getParamsPython(pybind11::tuple typ)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    validateTypes(typ1, typ2, "getting params");

    return m_params[m_typpair_idx(typ1, typ2)].toPython();
    }

template<class friction_evaluator>
void FrictionPotentialPair<friction_evaluator>::validateTypes(unsigned int typ1,
                                                        unsigned int typ2,
                                                        std::string action)
    {
    // TODO change logic to just throw an exception
    auto n_types = this->m_pdata->getNTypes();
    if (typ1 >= n_types || typ2 >= n_types)
        {
        throw std::runtime_error("Error in" + action + " for pair potential. Invalid type");
        }
    }

/*! \param typ The type index.
    \param param Shape parameter to set
          set.
*/
template<class friction_evaluator>
void FrictionPotentialPair<friction_evaluator>::setShape(unsigned int typ, const shape_type& shape_param)
    {
    if (typ >= m_pdata->getNTypes())
        {
        throw std::runtime_error("Error setting shape parameters in FrictionPotentialPair");
        }

    m_shape_params[typ] = shape_param;
    }

/*! \param typ The type index.
    \param param Shape parameter to set
          set.
*/
template<class friction_evaluator>
void FrictionPotentialPair<friction_evaluator>::setShapePython(std::string typ,
                                                         pybind11::object shape_param)
    {
    auto typ_ = m_pdata->getTypeByName(typ);
    setShape(typ_, shape_type(shape_param, m_exec_conf->isCUDAEnabled()));
    }

/*! \param typ The type index.
    \param param Shape parameter to set
          set.
*/
template<class friction_evaluator>
pybind11::object FrictionPotentialPair<friction_evaluator>::getShapePython(std::string typ)
    {
    auto typ_ = m_pdata->getTypeByName(typ);
    if (typ_ >= m_pdata->getNTypes())
        {
        throw std::runtime_error("Error getting shape parameters in FrictionPotentialPair");
        }

    return m_shape_params[typ_].toPython();
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param rcut Cutoff radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
template<class friction_evaluator>
void FrictionPotentialPair<friction_evaluator>::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    validateTypes(typ1, typ2, "setting r_cut");
        {
        // store r_cut**2 for use internally
        ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::readwrite);
        h_rcutsq.data[m_typpair_idx(typ1, typ2)] = rcut * rcut;
        h_rcutsq.data[m_typpair_idx(typ2, typ1)] = rcut * rcut;

        // store r_cut unmodified for so the neighbor list knows what particles to include
        ArrayHandle<Scalar> h_r_cut_nlist(*m_r_cut_nlist,
                                          access_location::host,
                                          access_mode::readwrite);
        h_r_cut_nlist.data[m_typpair_idx(typ1, typ2)] = rcut;
        h_r_cut_nlist.data[m_typpair_idx(typ2, typ1)] = rcut;
        }

    // notify the neighbor list that we have changed r_cut values
    m_nlist->notifyRCutMatrixChange();
    }

template<class friction_evaluator>
void FrictionPotentialPair<friction_evaluator>::setRCutPython(pybind11::tuple types, Scalar r_cut)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    setRcut(typ1, typ2, r_cut);
    }

template<class friction_evaluator>
Scalar FrictionPotentialPair<friction_evaluator>::getRCut(pybind11::tuple types)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "getting r_cut.");
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    return sqrt(h_rcutsq.data[m_typpair_idx(typ1, typ2)]);
    }

/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is
   called to ensure that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template<class friction_evaluator>
void FrictionPotentialPair<friction_evaluator>::computeForces(uint64_t timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);
    bool nlist_updated = m_nlist->hasBeenUpdated(timestep);

    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::host,
                                  access_mode::read);
    ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                   access_location::host,
                                   access_mode::read);

    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // force arrays
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    // let's handle the startup and rebuild case
    if (!m_dynamic_state_flag || nlist_updated)
        {
        m_dynamic_state_flag = true;

        std::vector<Scalar3, hoomd::detail::managed_allocator<Scalar3>> xi;
        std::vector<Scalar3, hoomd::detail::managed_allocator<Scalar3>> psi;
        std::unordered_map<std::pair<unsigned int, unsigned int>, unsigned int, pair_hash> pair_idx;

        // maybe we don't need to reserve?
        xi.reserve(m_xi.size());
        psi.reserve(m_psi.size());
        pair_idx.reserve(m_pair_idx.bucket_count());

        xi.swap(m_xi);
        psi.swap(m_psi);
        pair_idx.swap(m_pair_idx);

        clearDynamicState();

        unsigned int idx = 0;
        for (int i = 0; i < (int)m_pdata->getN(); i++)
            {
            auto tag_i = h_tag.data[i];
            const size_t myHead = h_head_list.data[i];
            const unsigned int size = (unsigned int)h_n_neigh.data[i];
            for (unsigned int k = 0; k < size; k++)
                {
                // access the index of this neighbor (MEM TRANSFER: 1 scalar)
                unsigned int j = h_nlist.data[myHead + k];
                auto tag_j = h_tag.data[j];
                auto pair = std::make_pair(tag_i, tag_j);
                if (pair_idx.empty() || pair_idx.find(pair) == pair_idx.end())
                    {
                    m_xi.push_back(make_scalar3(0, 0, 0));
                    m_psi.push_back(make_scalar3(0, 0, 0));
                    }
                else
                    {
                    unsigned int mapped_idx = pair_idx[pair];

                    m_xi.push_back(xi[mapped_idx]);
                    m_psi.push_back(psi[mapped_idx]);
                    }
                m_pair_idx[pair] = idx;
                idx++;
                }
            }
        }

    const BoxDim box = m_pdata->getBox();
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
        {
        // need to start from a zero force, energy and virial
        memset(&h_force.data[0], 0, sizeof(Scalar4) * m_pdata->getN());
        memset(&h_torque.data[0], 0, sizeof(Scalar4) * m_pdata->getN());
        memset(&h_virial.data[0], 0, sizeof(Scalar) * m_virial.getNumElements());

        PDataFlags flags = this->m_pdata->getFlags();
        bool compute_virial = flags[pdata_flag::pressure_tensor];

        auto xi_it = m_xi.begin();
        auto psi_it = m_psi.begin();

        // for each particle
        for (int i = 0; i < (int)m_pdata->getN(); i++)
            {
            // access the particle's position and type (MEM TRANSFER: 4 scalars)
            Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
            unsigned int typei = __scalar_as_int(h_pos.data[i].w);
            quat<Scalar> q_i(h_orientation.data[i]);
            quat<Scalar> p_i(h_angmom.data[i]);
            vec3<Scalar> v_i(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);
            vec3<Scalar> s_i((conj(q_i) * p_i).v / Scalar(2.0));
            Scalar3 I_i(h_inertia.data[i]);
            vec3<Scalar> w_i(
                I_i.x == 0.0 ? 0.0 : s_i.x / I_i.x,
                I_i.y == 0.0 ? 0.0 : s_i.y / I_i.y,
                I_i.z == 0.0 ? 0.0 : s_i.z / I_i.z
                );
            w_i = rotate(q_i, w_i); // now rotate into real frame

            // sanity check
            assert(typei < m_pdata->getNTypes());
            
            // access diameter and charge (if needed)
            Scalar di = Scalar(0.0);
            Scalar qi = Scalar(0.0);
            // if (friction_evaluator::needsDiameter())
            di = h_diameter.data[i];
            if (friction_evaluator::needsCharge())
                qi = h_charge.data[i];

            // initialize current particle force, torque, potential energy, and virial to 0
            Scalar fxi = Scalar(0.0);
            Scalar fyi = Scalar(0.0);
            Scalar fzi = Scalar(0.0);
            Scalar txi = Scalar(0.0);
            Scalar tyi = Scalar(0.0);
            Scalar tzi = Scalar(0.0);
            Scalar pei = Scalar(0.0);
            Scalar virialxxi = 0.0;
            Scalar virialxyi = 0.0;
            Scalar virialxzi = 0.0;
            Scalar virialyyi = 0.0;
            Scalar virialyzi = 0.0;
            Scalar virialzzi = 0.0;

            // loop over all of the neighbors of this particle
            const size_t myHead = h_head_list.data[i];
            const unsigned int size = (unsigned int)h_n_neigh.data[i];
            for (unsigned int k = 0; k < size; k++)
                {
                // access the index of this neighbor (MEM TRANSFER: 1 scalar)
                unsigned int j = h_nlist.data[myHead + k];
                assert(j < m_pdata->getN() + m_pdata->getNGhosts());

                // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
                Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
                Scalar3 dx = pi - pj;
                quat<Scalar> q_j(h_orientation.data[j]);
                quat<Scalar> p_j(h_angmom.data[j]);
                vec3<Scalar> v_j(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
                vec3<Scalar> s_j((conj(q_j) * p_j).v / Scalar(2.0));
                Scalar3 I_j(h_inertia.data[j]);
                vec3<Scalar> w_j(
                    I_j.x == 0.0 ? 0.0 : s_j.x / I_j.x,
                    I_j.y == 0.0 ? 0.0 : s_j.y / I_j.y,
                    I_j.z == 0.0 ? 0.0 : s_j.z / I_j.z
                    );
                w_j = rotate(q_j, w_j); // now rotate into real frame

                // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
                unsigned int typej = __scalar_as_int(h_pos.data[j].w);
                assert(typej < m_pdata->getNTypes());

                // access diameter and charge (if needed)
                Scalar dj = Scalar(0.0);
                Scalar qj = Scalar(0.0);
                // if (friction_evaluator::needsDiameter())
                dj = h_diameter.data[j];
                if (friction_evaluator::needsCharge())
                    qj = h_charge.data[j];

                // apply periodic boundary conditions
                dx = box.minImage(dx);
                vec3<Scalar> v_dx(dx);
                auto rsqr = dot(dx, dx);
                auto unit_dx = dx * fast::rsqrt(rsqr);


                // get parameters for this type pair
                unsigned int typpair_idx = m_typpair_idx(typei, typej);
                const param_type& param = m_params[typpair_idx];
                Scalar rcutsq = h_rcutsq.data[typpair_idx];

                // design specifies that energies are shifted if
                // shift mode is set to shift
                bool energy_shift = false;
                if (m_shift_mode == shift)
                    energy_shift = true;

                friction_evaluator eval(rsqr, rcutsq, param);

                if (friction_evaluator::needsDiameter())
                    eval.setDiameter(di, dj);
                if (friction_evaluator::needsCharge())
                    eval.setCharge(qi, qj);
                if (friction_evaluator::needsShape())
                    eval.setShape(&m_shape_params[typei], &m_shape_params[typej]);
                if (friction_evaluator::needsTags())
                    eval.setTags(h_tag.data[i], h_tag.data[j]);

                vec3<Scalar> xi_ij(*xi_it);
                vec3<Scalar> psi_ij(*psi_it);

                Scalar force_divr = Scalar(0.0);
                Scalar pair_eng = Scalar(0.0);

                // compute the force and potential energy
                vec3<Scalar> force(0.0, 0.0, 0.0);
                vec3<Scalar> force_slide(0.0, 0.0, 0.0);
                vec3<Scalar> force_roll(0.0, 0.0, 0.0);
                vec3<Scalar> torque_i(0.0, 0.0, 0.0);
                vec3<Scalar> torque_j(0.0, 0.0, 0.0);

                FrictionResult result = eval.evaluate(force_divr, pair_eng, energy_shift);

                bool evaluated = result.eval;
                bool contact = result.contact;

                force = force_divr * v_dx;

                if (contact)
                    {
                    Scalar mus = eval.mus;
                    Scalar mur = eval.mur;
                    Scalar ks = eval.ks;
                    Scalar kr = eval.kr;

                    Scalar force_sqr = force_divr * force_divr * rsqr;

                    vec3<Scalar> v_unit_dx(unit_dx.x, unit_dx.y, unit_dx.z);
                    
                    force_slide = ks * xi_ij;
                    force_roll = kr * psi_ij;
                    if (dot(force_slide, force_slide) > mus * force_sqr)
                        force_slide = mus * force;
                    if (dot(force_roll, force_roll) > mur * force_sqr)
                        force_roll = mur * force;

                    force += force_slide;

                    Scalar a_ij = Scalar(2.0) * di * dj / (di + dj);
                    vec3<Scalar> ur_pre = (v_j - v_i) - cross(di*w_i + dj*w_j, v_unit_dx);
                    auto tmp = ur_pre - v_unit_dx * dot(ur_pre, v_unit_dx) * m_deltaT;
                    *xi_it += make_scalar3(tmp.x, tmp.y, tmp.z);
                    tmp = a_ij * cross(w_i - w_j, v_unit_dx) * m_deltaT;
                    *psi_it += make_scalar3(tmp.x, tmp.y, tmp.z);

                    torque_i = di * cross(v_unit_dx, force_slide) + a_ij * cross(v_unit_dx, force_roll);
                    if (third_law)
                        torque_j = dj * cross(v_unit_dx, -force_slide) + a_ij * cross(v_unit_dx, -force_roll);
                    }
                else
                    {
                    *xi_it = make_scalar3(0.0, 0.0, 0.0);
                    *psi_it = make_scalar3(0.0, 0.0, 0.0);
                    }

                ++xi_it;
                ++psi_it;

                if (evaluated)
                    {
                    Scalar3 force2 = Scalar(0.5) * make_scalar3(force.x, force.y, force.z);

                    // add the force, potential energy and virial to the particle i
                    // (FLOPS: 8)
                    fxi += force.x;
                    fyi += force.y;
                    fzi += force.z;
                    txi += torque_i.x;
                    tyi += torque_i.y;
                    tzi += torque_i.z;
                    pei += pair_eng * Scalar(0.5);

                    if (compute_virial)
                        {
                        virialxxi += dx.x * force2.x;
                        virialxyi += dx.y * force2.x;
                        virialxzi += dx.z * force2.x;
                        virialyyi += dx.y * force2.y;
                        virialyzi += dx.z * force2.y;
                        virialzzi += dx.z * force2.z;
                        }

                    // add the force to particle j if we are using the third law (MEM TRANSFER: 10
                    // scalars / FLOPS: 8)
                    // assert(!third_law);
                    if (third_law)
                        {
                        h_force.data[j].x -= force.x;
                        h_force.data[j].y -= force.y;
                        h_force.data[j].z -= force.z;
                        h_torque.data[j].x -= torque_j.x;
                        h_torque.data[j].y -= torque_j.y;
                        h_torque.data[j].z -= torque_j.z;
                        h_force.data[j].w += pair_eng * Scalar(0.5);
                        if (compute_virial)
                            {
                            h_virial.data[0 * m_virial_pitch + j] += dx.x * force2.x;
                            h_virial.data[1 * m_virial_pitch + j] += dx.y * force2.x;
                            h_virial.data[2 * m_virial_pitch + j] += dx.z * force2.x;
                            h_virial.data[3 * m_virial_pitch + j] += dx.y * force2.y;
                            h_virial.data[4 * m_virial_pitch + j] += dx.z * force2.y;
                            h_virial.data[5 * m_virial_pitch + j] += dx.z * force2.z;
                            }
                        }
                    }
                }

            // finally, increment the force, potential energy and virial for particle i
            h_force.data[i].x += fxi;
            h_force.data[i].y += fyi;
            h_force.data[i].z += fzi;
            h_torque.data[i].x += txi;
            h_torque.data[i].y += tyi;
            h_torque.data[i].z += tzi;
            h_force.data[i].w += pei;
            if (compute_virial)
                {
                h_virial.data[0 * m_virial_pitch + i] += virialxxi;
                h_virial.data[1 * m_virial_pitch + i] += virialxyi;
                h_virial.data[2 * m_virial_pitch + i] += virialxzi;
                h_virial.data[3 * m_virial_pitch + i] += virialyyi;
                h_virial.data[4 * m_virial_pitch + i] += virialyzi;
                h_virial.data[5 * m_virial_pitch + i] += virialzzi;
                }
            }
        }
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template<class friction_evaluator>
CommFlags FrictionPotentialPair<friction_evaluator>::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = CommFlags(0);

    // we need orientations for anisotropic ptls
    flags[comm_flag::orientation] = 1;

    if (friction_evaluator::needsCharge())
        flags[comm_flag::charge] = 1;

    if (friction_evaluator::needsDiameter())
        flags[comm_flag::diameter] = 1;

    // with rigid bodies, include net torque
    flags[comm_flag::net_torque] = 1;

    flags |= ForceCompute::getRequestedCommFlags(timestep);

    return flags;
    }
#endif

namespace detail
    {

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_FrictionPotentialPair(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<FrictionPotentialPair<T>, ForceCompute, std::shared_ptr<FrictionPotentialPair<T>>>
        pairpotential(m, name.c_str());
    pairpotential
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("setParams", &FrictionPotentialPair<T>::setParamsPython)
        .def("getParams", &FrictionPotentialPair<T>::getParamsPython)
        .def("setShape", &FrictionPotentialPair<T>::setShapePython)
        .def("getShape", &FrictionPotentialPair<T>::getShapePython)
        .def("setRCut", &FrictionPotentialPair<T>::setRCutPython)
        .def("getRCut", &FrictionPotentialPair<T>::getRCut)
        .def_property("mode",
                      &FrictionPotentialPair<T>::getShiftMode,
                      &FrictionPotentialPair<T>::setShiftModePython)
        .def("slotWriteGSDShapeSpec", &FrictionPotentialPair<T>::slotWriteGSDShapeSpec)
        .def("connectGSDShapeSpec", &FrictionPotentialPair<T>::connectGSDShapeSpec);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // __FRICTION_POTENTIAL_PAIR_H__
