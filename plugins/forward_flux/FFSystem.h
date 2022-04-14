
#include "hoomd/System.h"

#include <types.h>
#include <optional>

#ifndef __FF_SYSTEM_H__
#define __FF_SYSTEM_H__

/*! \file ff_system.h
    \brief Declares the Forward Flux System class and associated helper classes
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {

/** In order to do this all rhobustly in MD, I think we need to constain this to
    methods where we can apply FIRE (or GD)
*/
class PYBIND11_EXPORT FFSystem: public System
    {
    public:
    //! Constructor
    FFSystem(std::shared_ptr<SystemDefinition> sysdef, uint64_t initial_tstep);

    //! Run the forward flux calculation segment, until the op
    void runFFTrial(const double barrier, const SystemDefinition& sysdef);

    void sampleBasin();

    private:

    double(*orderParam)(SystemDefinition*);
    bool quenched;
    // bool isChild; // Maybe later we want to try some simple parallelism with this?

    //! 
    void runFIRE();
    
    };

namespace detail
    {

void export_FFSystem(pybind11::module& m);

    } // end namespace detail
    } // end namespace hoomd

#endif // __FF_SYSTEM_H__