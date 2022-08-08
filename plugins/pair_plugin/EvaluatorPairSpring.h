// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

#ifndef __PAIR_EVALUATOR_SPRING_H__
#define __PAIR_EVALUATOR_SPRING_H__

#ifndef __HIPCC__
#include <string>
#include <optional>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairSpring.h
    \brief Defines the pair evaluator class for the harmonic spring potential
    \details Harmonic Spring potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {

class EvaluatorPairHarmonic
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar k;
        Scalar rcut;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

#ifndef __HIPCC__
        param_type() : k(0), rcut(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            auto _k(v["k"].cast<Scalar>());
            auto _rcut(v["rcut"].cast<Scalar>());
            k = _k;
            rcut = _rcut;
            }

        // this constructor facilitates unit testing
        param_type(Scalar k, Scalar rcut, bool managed = false)
            {
            this->k = k;
            this->rcut = rcut;
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["k"] = k;
            v["rcut"] = rcut;
            return v;
            }
#endif
        }
#ifdef SINGLE_PRECISION
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairHarmonic(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), k(_params.k), rcut(_params.rcut)
        {
        }

    //! We use precomputed rcut in place of di + dj for performance
    DEVICE static bool needsDiameter()
        {
        return false;
        }
    //! Accept the optional diameter values
    /*! \param di Diameter of particle i
        \param dj Diameter of particle j
    */
    DEVICE void setDiameter(Scalar di, Scalar dj) { }

    //! Harmonic doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }
    //! Accept the optional diameter values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param pair_eng Output parameter to write the computed pair energy
        \param energy_shift If true, the potential must be shifted so that
        V(r) is continuous at the cutoff
        \note There is no need to check if rsq < rcutsq in this method.
        Cutoff tests are performed in PotentialPair.

        \return True if they are evaluated or false if they are not because
        we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        // compute the force divided by r in force_divr
        if (rsq < rcutsq && k != 0)
            {
            
            Scalar r = fast::sqrt(rsq);
            Scalar rinv = Scalar(1.0) / r;
            Scalar term = rcut - r;

            
            force_divr = k * rinv * term;

            pair_eng =  Scalar(0.5) * k * term * term;

            return true;
            }
        else
            return false;
        }

    /*! Specialization of the force/energy evaluator for the hard particle friction compute.

        Saves at least one repeated sqrt and division that is already needed for the HPF 
        force compute.
    */
    DEVICE bool evalForceAndEnergyHPF(Scalar& force_divr, Scalar& pair_eng, Scalar& r, Scalar& rinv)
        {
        // compute the force divided by r in force_divr
        if (rsq < rcutsq && k != 0)
            {
            
            r = fast::sqrt(rsq);
            rinv = Scalar(1.0) / r;
            Scalar term = rcut - r;

            
            force_divr = k * rinv * term;

            pair_eng =  Scalar(0.5) * k * term * term;

            return true;
            }
        else
            return false;
        }

    DEVICE Scalar evalPressureLRCIntegral()
        {
        return 0;
        }

    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return 0;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("spring");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;    //!< Stored rsq from the constructor
    Scalar rcutsq; //!< Stored rcutsq from the constructor
    Scalar k;      //!< spring constant
    Scalar rcut;   //!< contact distance
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_SPRING_H__
