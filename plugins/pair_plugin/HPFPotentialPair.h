// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __HPF_POTENTIAL_PAIR_H__
#define __HPF_POTENTIAL_PAIR_H__

#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "hoomd/md/NeighborList.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/GSDShapeSpecWriter.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/managed_allocator.h"
#include "hoomd/md/EvaluatorPairLJ.h"

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

/*! \file HPFPotentialPair.h
    \brief Defines the template class for the hard-particle frictional interaction
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {

//! Hash function for std::pair
struct pair_hash
    {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const
        {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
        }
    };

//! Template class for computing pair potentials
/*! <b>Overview:</b>
    HPFPotentialPair computes standard pair potentials (and forces) between all particle pairs in the
   simulation. It employs the use of a neighbor list to limit the number of computations done to
   only those particles with the cutoff radius of each other. The computation of the actual V(r) is
   not performed directly by this class, but by an evaluator class (e.g. EvaluatorPairSpring) which is
   passed in as a template parameter so the computations are performed as efficiently as possible.

    <b>Implementation details</b>

    rcutsq, and the params are stored per particle type pair. It wastes a little bit of
   space, but benchmarks show that storing the symmetric type pairs and indexing with Index2D is
   faster than not storing redundant pairs and indexing with Index2DUpperTriangular. All of these
   values are stored in GlobalArray for easy access on the GPU by a derived class. The type of the
   parameters is defined by \a param_type in the potential evaluator class passed in. See the
   appropriate documentation for the evaluator for the definition of each element of the parameters.
*/
template<class evaluator> class HPFPotentialPair : public ForceCompute
    {
    public:
    //! Param type from evaluator
    typedef typename evaluator::param_type param_type;

    //! Construct the pair potential
    HPFPotentialPair(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<NeighborList> nlist,
                     Scalar mus, Scalar mur, Scalar ks, Scalar kr);
    //! Destructor
    virtual ~HPFPotentialPair();

    //! Set and get the pair parameters for a single type pair
    virtual void setParams(unsigned int typ1, unsigned int typ2, const param_type& param);
    virtual void setParamsPython(pybind11::tuple typ, pybind11::dict params);
    /// Get params for a single type pair using a tuple of strings
    virtual pybind11::dict getParams(pybind11::tuple typ);
    //! Set the rcut for a single type pair
    virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);
    /// Get the r_cut for a single type pair
    Scalar getRCut(pybind11::tuple types);
    /// Set the rcut for a single type pair using a tuple of strings
    virtual void setRCutPython(pybind11::tuple types, Scalar r_cut);
    //! Method that is called whenever the GSD file is written if connected to a GSD file.
    int slotWriteGSDShapeSpec(gsd_handle&) const;
    /// Validate that types are within Ntypes
    void validateTypes(unsigned int typ1, unsigned int typ2, std::string action);
    //! Method that is called to connect to the gsd write state signal
    void connectGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer);

    // It might be better to just remove these shift methods, since we don't
    // plan to use them with this force compute
    //! Shifting modes that can be applied to the energy
    enum energyShiftMode
        {
        no_shift = 0
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
        else
            {
            throw std::runtime_error("Invalid energy shift mode.");
            }
        }

    /// Get the mode used for the energy shifting
    std::string getShiftMode()
        {
        switch (m_shift_mode)
            {
        case no_shift:
            return "none";
        default:
            throw std::runtime_error("Error setting shift mode.");
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

    std::vector<std::string> getTypeShapeMapping() const
        {
        std::vector<std::string> type_shape_mapping(m_pdata->getNTypes());
        for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
            {
            evaluator eval(Scalar(0.0), Scalar(0.0), this->m_params[m_typpair_idx(i, i)]);
            type_shape_mapping[i] = eval.getShapeSpec();
            }
        return type_shape_mapping;
        }

    bool m_log_pair_info = false;
    Scalar m_gamma = Scalar(0.0); 

    protected:
    std::shared_ptr<NeighborList> m_nlist; //!< The neighborlist to use for the computation
    energyShiftMode m_shift_mode; //!< Store the mode with which to handle the energy shift at r_cut
    Index2D m_typpair_idx;        //!< Helper class for indexing per type pair arrays
    GlobalArray<Scalar> m_rcutsq; //!< Cutoff radius squared per type pair

    /// Per type pair potential parameters
    std::vector<param_type, hoomd::detail::managed_allocator<param_type>> m_params;

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

    std::unordered_map<unsigned int, Scalar3> m_w_cache; //!< Cache for the angular velocity of each particle

    /// r_cut (not squared) given to the neighbor list
    std::shared_ptr<GlobalArray<Scalar>> m_r_cut_nlist;

    /// Keep track of number of each type of particle
    std::vector<unsigned int> m_num_particles_by_type;

    Scalar m_mus = Scalar(0.0); //!< Sliding friction coefficient
    Scalar m_mur = Scalar(0.0); //!< Rolling friction coefficient
    Scalar m_ks = Scalar(10.0); //!< Sliding friction spring constant
    Scalar m_kr = Scalar(10.0); //!< Rolling friction spring constant

#ifdef ENABLE_MPI
    /// The system's communicator.
    std::shared_ptr<Communicator> m_comm;
#endif

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    }; // end class HPFPotentialPair

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
*/
template<class evaluator>
HPFPotentialPair<evaluator>::HPFPotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                                        std::shared_ptr<NeighborList> nlist,
                                        Scalar mus,
                                        Scalar mur,
                                        Scalar ks,
                                        Scalar kr)
    : ForceCompute(sysdef), m_nlist(nlist), m_shift_mode(no_shift),
      m_typpair_idx(m_pdata->getNTypes()), m_mus(mus), m_mur(mur), m_ks(ks), m_kr(kr)
    {
    m_exec_conf->msg->notice(5) << "Constructing HPFPotentialPair<" << evaluator::getName() << ">"
                                << std::endl;

    assert(m_pdata);
    assert(m_nlist);

    // initialize empty containers for xi, psi, pair_idx, and w_cache
    std::vector<Scalar3, hoomd::detail::managed_allocator<Scalar3>> xi;
    std::vector<Scalar3, hoomd::detail::managed_allocator<Scalar3>> psi;
    std::unordered_map<std::pair<unsigned int, unsigned int>, unsigned int, pair_hash> pair_idx;
    std::unordered_map<unsigned int, Scalar3> w_cache;

    auto num_nlist_elements = 0;
    xi.reserve(num_nlist_elements);
    psi.reserve(num_nlist_elements);
    pair_idx.reserve(m_pair_idx.max_load_factor()*num_nlist_elements);
    w_cache.reserve(w_cache.max_load_factor()*num_nlist_elements);

    m_xi.swap(xi);
    m_psi.swap(psi);
    m_pair_idx.swap(pair_idx);
    m_w_cache.swap(w_cache);

    GlobalArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_rcutsq.swap(rcutsq);
    m_params = std::vector<param_type, hoomd::detail::managed_allocator<param_type>>(
        m_typpair_idx.getNumElements(),
        param_type(),
        hoomd::detail::managed_allocator<param_type>(m_exec_conf->isCUDAEnabled()));

    m_r_cut_nlist
        = std::make_shared<GlobalArray<Scalar>>(m_typpair_idx.getNumElements(), m_exec_conf);
    nlist->addRCutMatrix(m_r_cut_nlist);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_pdata->getExecConf()->isCUDAEnabled())
        {
        // m_params is _always_ in unified memory, so memadvise and prefetch
        cudaMemAdvise(m_params.data(),
                      m_params.size() * sizeof(param_type),
                      cudaMemAdviseSetReadMostly,
                      0);
        auto& gpu_map = m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemPrefetchAsync(m_params.data(),
                                 sizeof(param_type) * m_params.size(),
                                 gpu_map[idev]);
            }

        // m_rcutsq and m_ronsq only in unified memory if allConcurrentManagedAccess
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

    // get number of each type of particle, needed for energy and pressure correction
    m_num_particles_by_type.resize(m_pdata->getNTypes());
    std::fill(m_num_particles_by_type.begin(), m_num_particles_by_type.end(), 0);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int typeid_i = __scalar_as_int(h_postype.data[i].w);
        m_num_particles_by_type[typeid_i] += 1;
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // reduce number of each type of particle on all processors
        MPI_Allreduce(MPI_IN_PLACE,
                      m_num_particles_by_type.data(),
                      m_pdata->getNTypes(),
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm_weak = m_sysdef->getCommunicator();
        assert(comm_weak.lock());
        m_comm = comm_weak.lock();
        }
#endif
    }

template<class evaluator> HPFPotentialPair<evaluator>::~HPFPotentialPair()
    {
    m_exec_conf->msg->notice(5) << "Destroying HPFPotentialPair<" << evaluator::getName() << ">"
                                << std::endl;

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
template<class evaluator>
void HPFPotentialPair<evaluator>::setParams(unsigned int typ1,
                                         unsigned int typ2,
                                         const param_type& param)
    {
    validateTypes(typ1, typ2, "setting params");
    m_params[m_typpair_idx(typ1, typ2)] = param;
    m_params[m_typpair_idx(typ2, typ1)] = param;
    }

template<class evaluator>
void HPFPotentialPair<evaluator>::setParamsPython(pybind11::tuple typ, pybind11::dict params)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    setParams(typ1, typ2, param_type(params, m_exec_conf->isCUDAEnabled()));
    }

template<class evaluator> pybind11::dict HPFPotentialPair<evaluator>::getParams(pybind11::tuple typ)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    validateTypes(typ1, typ2, "setting params");

    return m_params[m_typpair_idx(typ1, typ2)].asDict();
    }

template<class evaluator>
void HPFPotentialPair<evaluator>::validateTypes(unsigned int typ1,
                                             unsigned int typ2,
                                             std::string action)
    {
    auto n_types = this->m_pdata->getNTypes();
    if (typ1 >= n_types || typ2 >= n_types)
        {
        throw std::runtime_error("Error in" + action + " for pair potential. Invalid type");
        }
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param rcut Cutoff radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
template<class evaluator>
void HPFPotentialPair<evaluator>::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
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

template<class evaluator>
void HPFPotentialPair<evaluator>::setRCutPython(pybind11::tuple types, Scalar r_cut)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    setRcut(typ1, typ2, r_cut);
    }

template<class evaluator> Scalar HPFPotentialPair<evaluator>::getRCut(pybind11::tuple types)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "getting r_cut.");
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    return sqrt(h_rcutsq.data[m_typpair_idx(typ1, typ2)]);
    }

template<class evaluator>
void HPFPotentialPair<evaluator>::connectGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer)
    {
    typedef hoomd::detail::SharedSignalSlot<int(gsd_handle&)> SlotType;
    auto func
        = std::bind(&HPFPotentialPair<evaluator>::slotWriteGSDShapeSpec, this, std::placeholders::_1);
    std::shared_ptr<hoomd::detail::SignalSlot> pslot(new SlotType(writer->getWriteSignal(), func));
    addSlot(pslot);
    }

template<class evaluator>
int HPFPotentialPair<evaluator>::slotWriteGSDShapeSpec(gsd_handle& handle) const
    {
    hoomd::detail::GSDShapeSpecWriter shapespec(m_exec_conf);
    m_exec_conf->msg->notice(10) << "HPFPotentialPair writing to GSD File to name: "
                                 << shapespec.getName() << std::endl;
    int retval = shapespec.write(handle, this->getTypeShapeMapping());
    return retval;
    }

/*! \post The pair forces are computed for the given timestep. The neighborlist's compute method is
   called to ensure that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template<class evaluator> void HPFPotentialPair<evaluator>::computeForces(uint64_t timestep)
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

    const BoxDim box = m_pdata->getGlobalBox();
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    // need to start from a zero force, energy and virial
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_torque.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

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

    auto xi_it = m_xi.begin();
    auto psi_it = m_psi.begin();

    m_w_cache.clear(); // clear angular velocity cache for the upcoming run

    // for each particle
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);        

        vec3<Scalar> v_i(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);
        vec3<Scalar> w_i(0.0, 0.0, 0.0);
        auto tag_i = h_tag.data[i];

        auto search = m_w_cache.find(tag_i);
        if (search == m_w_cache.end())
            {
            quat<Scalar> q_i(h_orientation.data[i]);
            quat<Scalar> p_i(h_angmom.data[i]);
            
            Scalar3 I_i(h_inertia.data[i]);
            vec3<Scalar> s_i((conj(q_i) * p_i).v / Scalar(2.0));
            // I might be able to get rid of the ? operator
            // this would assume that any componenet of s_i is always zero if I_i is zero
            w_i = vec3<Scalar>(
                I_i.x == 0.0 ? 0.0 : s_i.x / I_i.x,
                I_i.y == 0.0 ? 0.0 : s_i.y / I_i.y,
                I_i.z == 0.0 ? 0.0 : s_i.z / I_i.z
            );
            w_i = rotate(q_i, w_i); // now rotate into real frame
            m_w_cache[tag_i] = make_scalar3(w_i.x, w_i.y, w_i.z);
            }
        else
            {
            Scalar3 _w = search->second;
            w_i = vec3<Scalar>(_w.x, _w.y, _w.z);
            }
        

        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access diameter and charge (if needed)
        Scalar di = Scalar(0.0);
        Scalar qi = Scalar(0.0);
        // if (evaluator::needsDiameter())
        di = Scalar(0.5) * h_diameter.data[i];
        if (evaluator::needsCharge())
            qi = h_charge.data[i];

        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar3 ti = make_scalar3(0, 0, 0);
        Scalar pei = 0.0;
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

            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar)
            unsigned int typej = __scalar_as_int(h_pos.data[j].w);
            assert(typej < m_pdata->getNTypes());

            // access diameter and charge (if needed)
            Scalar dj = Scalar(0.0);
            Scalar qj = Scalar(0.0);
            // if (evaluator::needsDiameter())
            dj = Scalar(0.5) * h_diameter.data[j];
            if (evaluator::needsCharge())
                qj = h_charge.data[j];

            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // calculate r_ij squared (FLOPS: 5)
            Scalar rsq = dot(dx, dx);

            // get parameters for this type pair
            unsigned int typpair_idx = m_typpair_idx(typei, typej);
            param_type param = m_params[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];

            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);
            Scalar r = Scalar(0.0);
            Scalar rinv = Scalar(0.0);
            evaluator eval(rsq, rcutsq, param);
            if (evaluator::needsDiameter())
                eval.setDiameter(di, dj);
            if (evaluator::needsCharge())
                eval.setCharge(qi, qj);
            
            //! This is the normal force of the conservative pair interaction.
            //! We'll also need to calculate the non-conservative friction forces
            //! if the conservative interaction is non-zero (in contact).
            bool evaluated = eval.evalForceAndEnergyHPF(force_divr, pair_eng, r, rinv);

            if (evaluated)
                {

                // grab data from particle j and calculate it if it isn't cached
                vec3<Scalar> v_j(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
                vec3<Scalar> w_j(0.0, 0.0, 0.0);
                auto tag_j = h_tag.data[j];

                //!> NOTE - eventually we probably want to avoid some of these computations if mus or mur are zero

                auto search = m_w_cache.find(tag_j);
                if (search == m_w_cache.end())
                    {
                    quat<Scalar> q_j(h_orientation.data[j]);
                    quat<Scalar> p_j(h_angmom.data[j]);
                    Scalar3 I_j(h_inertia.data[j]);
                    vec3<Scalar> s_j((conj(q_j) * p_j).v / Scalar(2.0));
                    // I might be able to get rid of the ? operator
                    // this would assume that any componenet of s_i is always zero if I_i is zero
                    w_j = vec3<Scalar>(
                        I_j.x == 0.0 ? 0.0 : s_j.x / I_j.x,
                        I_j.y == 0.0 ? 0.0 : s_j.y / I_j.y,
                        I_j.z == 0.0 ? 0.0 : s_j.z / I_j.z
                    );
                    w_j = rotate(q_j, w_i); // now rotate into real frame
                    m_w_cache[tag_j] = make_scalar3(w_j.x, w_j.y, w_j.z);
                    }
                else
                    {
                    Scalar3 _w = search->second;
                    w_j = vec3<Scalar>(_w.x, _w.y, _w.z);
                    }

                // add up conservate and non-conservative forces and compute torque
                vec3<Scalar> force = force_divr * vec3<Scalar>(dx.x, dx.y, dx.z);
                vec3<Scalar> force_slide(0.0, 0.0, 0.0);
                vec3<Scalar> force_roll(0.0, 0.0, 0.0);
                vec3<Scalar> torque_i(0.0, 0.0, 0.0);
                vec3<Scalar> torque_j(0.0, 0.0, 0.0);

                vec3<Scalar> xi_ij(*xi_it);
                vec3<Scalar> psi_ij(*psi_it);

                // non-conservative force and torque
                Scalar force_sqr = force_divr * force_divr * rsq;
                vec3<Scalar> v_unit_dx(dx.x*rinv, dx.y*rinv, dx.z*rinv);

                force_slide = m_ks * xi_ij;
                force_roll = m_kr * psi_ij;
                Scalar slide_sqr = dot(force_slide, force_slide);
                Scalar roll_sqr = dot(force_roll, force_roll);
                if (slide_sqr > m_mus * m_mus * force_sqr)
                    force_slide *= m_mus * fast::rsqrt(slide_sqr) * force_divr * r;
                if (roll_sqr > m_mur * m_mur * force_sqr)
                    force_roll *= m_mur * fast::rsqrt(roll_sqr) * force_divr * r;

                force += force_slide;

                Scalar a_ij = Scalar(2.0) * di * dj / (di + dj);
                vec3<Scalar> ur_pre = (v_j - v_i) - cross(di*w_i + dj*w_j, v_unit_dx);
                auto tmp = ur_pre - v_unit_dx * dot(ur_pre, v_unit_dx) * m_deltaT;
                *xi_it += make_scalar3(tmp.x, tmp.y, tmp.z);
                tmp = a_ij * cross(w_i - w_j, v_unit_dx) * m_deltaT;
                *psi_it += make_scalar3(tmp.x, tmp.y, tmp.z);

                auto torque_slide = cross(v_unit_dx, force_slide);
                auto torque_roll = cross(v_unit_dx, force_roll);
                torque_i = di * torque_slide + a_ij * torque_roll;

                ti.x += torque_i.x;
                ti.y += torque_i.y;
                ti.z += torque_i.z;

                // TODO need to verify that this is the correct, but since we assume the bodies are spherical,
                // their torques should just as simple as negating the force and multiplying by the other radius
                if (third_law) 
                    torque_j = - dj * torque_slide - a_ij * torque_roll;
                
                
                Scalar3 force2 = make_scalar3(force.x, force.y, force.z) * Scalar(0.5);
                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fi += make_scalar3(force.x, force.y, force.z);
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
                // scalars / FLOPS: 8) only add force to local particles
                if (third_law && j < m_pdata->getN())
                    {
                    unsigned int mem_idx = j;
                    h_force.data[mem_idx].x -= force.x;
                    h_force.data[mem_idx].y -= force.y;
                    h_force.data[mem_idx].z -= force.z;
                    // might be a bug, buth this should probably be +, not - signs
                    h_torque.data[j].x += torque_j.x;
                    h_torque.data[j].y += torque_j.y;
                    h_torque.data[j].z += torque_j.z;
                    h_force.data[mem_idx].w += pair_eng * Scalar(0.5);
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
            else
                {
                *xi_it = make_scalar3(0.0, 0.0, 0.0);
                *psi_it = make_scalar3(0.0, 0.0, 0.0);
                }

            ++xi_it;
            ++psi_it;
            }

        if (m_gamma != 0.0)
            {
            fi.x -= v_i.x * m_gamma;
            fi.y -= v_i.y * m_gamma;
            fi.z -= v_i.z * m_gamma;
            ti.x -= w_i.x * m_gamma;
            ti.y -= w_i.y * m_gamma;
            ti.z -= w_i.z * m_gamma;
            }

        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = i;
        h_force.data[mem_idx].x += fi.x;
        h_force.data[mem_idx].y += fi.y;
        h_force.data[mem_idx].z += fi.z;
        h_torque.data[mem_idx].x += ti.x;
        h_torque.data[mem_idx].y += ti.y;
        h_torque.data[mem_idx].z += ti.z;
        h_force.data[mem_idx].w += pei;
        if (compute_virial)
            {
            h_virial.data[0 * m_virial_pitch + mem_idx] += virialxxi;
            h_virial.data[1 * m_virial_pitch + mem_idx] += virialxyi;
            h_virial.data[2 * m_virial_pitch + mem_idx] += virialxzi;
            h_virial.data[3 * m_virial_pitch + mem_idx] += virialyyi;
            h_virial.data[4 * m_virial_pitch + mem_idx] += virialyzi;
            h_virial.data[5 * m_virial_pitch + mem_idx] += virialzzi;
            }
        }
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template<class evaluator>
CommFlags HPFPotentialPair<evaluator>::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = CommFlags(0);

    if (evaluator::needsCharge())
        flags[comm_flag::charge] = 1;

    if (evaluator::needsDiameter())
        flags[comm_flag::diameter] = 1;

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
template<class T> void export_HPFPotentialPair(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<HPFPotentialPair<T>, ForceCompute, std::shared_ptr<HPFPotentialPair<T>>>
        potentialpair(m, name.c_str());
    potentialpair
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, Scalar, Scalar, Scalar, Scalar>())
        .def("setParams", &HPFPotentialPair<T>::setParamsPython)
        .def("getParams", &HPFPotentialPair<T>::getParams)
        .def("setRCut", &HPFPotentialPair<T>::setRCutPython)
        .def("getRCut", &HPFPotentialPair<T>::getRCut)
        // .def("_evaluate", &HPFPotentialPair<T>::evaluate)
        .def("slotWriteGSDShapeSpec", &HPFPotentialPair<T>::slotWriteGSDShapeSpec)
        .def("connectGSDShapeSpec", &HPFPotentialPair<T>::connectGSDShapeSpec)
        .def_readwrite("log_pair_info", &HPFPotentialPair<T>::m_log_pair_info)
        .def_readwrite("gamma", &HPFPotentialPair<T>::m_gamma);

    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // __HPF_POTENTIAL_PAIR_H__
