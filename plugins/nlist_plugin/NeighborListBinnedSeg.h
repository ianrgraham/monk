// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/NeighborList.h"
#include "CellListSeg.h"

/*! \file NeighborListBinnedSeg.h
    \brief Declares the NeighborListBinnedSeg class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __NeighborListBinnedSeg_H__
#define __NeighborListBinnedSeg_H__

namespace hoomd
    {
namespace md
    {
//! Efficient neighbor list build on the CPU
/*! Implements the O(N) neighbor list build on the CPU using a cell list.

    \ingroup computes
*/
class PYBIND11_EXPORT NeighborListBinnedSeg : public NeighborList
    {
    public:
    //! Constructs the compute
    NeighborListBinnedSeg(std::shared_ptr<SystemDefinition> sysdef, Scalar r_buff, std::vector<std::pair<unsigned int, unsigned int>> segments);

    //! Destructor
    virtual ~NeighborListBinnedSeg();

    /// Notify NeighborList that a r_cut matrix value has changed
    virtual void notifyRCutMatrixChange()
        {
        m_update_cell_size = true;
        NeighborList::notifyRCutMatrixChange();
        }

    /// Make the neighborlist deterministic
    void setDeterministic(bool deterministic)
        {
        for (auto m_cl : this->m_cls)
            {
            m_cl->setSortCellList(deterministic);
            }
        }

    /// Get the deterministic flag
    bool getDeterministic()
        {
        bool result = true;
        for (auto m_cl : this->m_cls)
            {
            result &= m_cl->getSortCellList();
            }
        return result;
        }

    protected:
    std::vector<std::shared_ptr<CellListSeg>> m_cls; //!< The cell list

    /// Track when the cell size needs to be updated
    bool m_update_cell_size = true;

    //! Builds the neighbor list
    virtual void buildNlist(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
