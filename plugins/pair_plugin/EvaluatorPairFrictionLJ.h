// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __EVALUATOR_PAIR_FRICTION_LJ_H__
#define __EVALUATOR_PAIR_FRICTION_LJ_H__

#ifndef __HIPCC__
#include <string>
#endif

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include "hoomd/VectorMath.h"
#include "FrictionPotentialPair.h"

/*! \file EvaluatorPairFrictionLJ.h
    \brief Defines a an evaluator class for the frictional LJ potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host
//! compiler
#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE
#endif


namespace hoomd
    {
namespace md
    {

/*!
 * Frictional LJ potential
 */

class EvaluatorPairFrictionLJ
    {
    public:
    struct param_type
        {
        Scalar epsilon; //! The energy scale.
        Scalar aes;     //! Attractive energy scaling
        Scalar sigma;   //! Interaction length scale
        Scalar lj1;
        Scalar lj2;
        Scalar dlt;
        Scalar mus;     //! Slding friction coefficient
        Scalar mur;     //! Rolling friction coefficient
        Scalar ks;      //! Sliding stiffness
        Scalar kr;      //! Rolling stiffness

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory
            allocation
        */
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

        HOSTDEVICE param_type()
            {
            epsilon = 0;
            aes = 0;
            sigma = 0;
            lj1 = 0;
            lj2 = 0;
            dlt = 0;
            mus = 0;
            mur = 0;
            ks = 0;
            kr = 0;
            }

#ifndef __HIPCC__

        param_type(pybind11::dict v, bool managed = false)
            {
            auto sigma(v["sigma"].cast<Scalar>());
            auto epsilon(v["epsilon"].cast<Scalar>());
            auto delta(v["delta"].cast<Scalar>());
            this->epsilon = v["epsilon"].cast<Scalar>();
            this->sigma = v["sigma"].cast<Scalar>();
            auto dsigma = sigma * (1.0 - delta / pow(2.0, 1. / 6.));
            aes = v["aes"].cast<Scalar>();
            lj1 = 4.0 * epsilon * pow(dsigma, 12.0);
            lj2 = 4.0 * epsilon * pow(dsigma, 6.0);
            dlt = delta;
            mus = v["mus"].cast<Scalar>();
            mur = v["mur"].cast<Scalar>();
            ks = v["ks"].cast<Scalar>();
            kr = v["kr"].cast<Scalar>();
            }

        pybind11::dict toPython()
            {
            pybind11::dict v;
            v["epsilon"] = epsilon;
            v["aes"] = aes;
            v["sigma"] = sigma;
            v["delta"] = dlt;
            v["mus"] = mus;
            v["mur"] = mur;
            v["ks"] = ks;
            v["kr"] = kr;
            return v;
            }

#endif
        }
#ifdef SINGLE_PRECISION
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    // Nullary structure required by FrictionPotentialPair.
    struct shape_type
        {
        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory allocation
        */
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

        HOSTDEVICE shape_type() { }

#ifndef __HIPCC__

        shape_type(pybind11::object shape_params, bool managed) { }

        pybind11::object toPython()
            {
            return pybind11::none();
            }
#endif

#ifdef ENABLE_HIP
        //! Attach managed memory to CUDA stream
        void set_memory_hint() const { }
#endif
        };

    //! Constructs the pair potential evaluator
    /*! \param _dr Displacement vector between particle centers of mass
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _q_i Quaternion of i^th particle
        \param _q_j Quaternion of j^th particle
        \param _params Per type pair parameters of this potential
    */
    HOSTDEVICE EvaluatorPairFrictionLJ(const Scalar _rsqr,
                                       const Scalar _rcutsq,
                                       const param_type& _params)
        : rsqr(_rsqr), rcutsq(_rcutsq), lj1(_params.lj1),
          lj2(_params.lj2), epsilon(_params.epsilon), aes(_params.aes), sigma(_params.sigma), dlt(_params.dlt), mus(_params.mus), mur(_params.mur), ks(_params.ks), kr(_params.kr)
        {
        }

    //! uses diameter
    HOSTDEVICE static bool needsDiameter()
        {
        return true;
        }

    //! Whether the pair potential uses shape.
    HOSTDEVICE static bool needsShape()
        {
        return false;
        }

    //! Whether the pair potential needs particle tags.
    HOSTDEVICE static bool needsTags()
        {
        return false;
        }

    //! whether pair potential requires charges
    HOSTDEVICE static bool needsCharge()
        {
        return false;
        }

    /// Whether the potential implements the energy_shift parameter
    HOSTDEVICE static bool constexpr implementsEnergyShift()
        {
        return true;
        }

    //! Accept the optional diameter values
    /*! \param di Diameter of particle i
        \param dj Diameter of particle j
    */
    HOSTDEVICE void setDiameter(Scalar di, Scalar dj) {
        auto ave_diam = (di + dj) / Scalar(2.0);
        delta = ave_diam - Scalar(1.0);
        // ai = di;
        // aj = dj;
        // aij = ai * aj / ave_diam;
    }

    //! Accept the optional shape values
    /*! \param shape_i Shape of particle i
        \param shape_j Shape of particle j
    */
    HOSTDEVICE void setShape(const shape_type* shapei, const shape_type* shapej) { }

    //! Accept the optional tags
    /*! \param tag_i Tag of particle i
        \param tag_j Tag of particle j
    */
    HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj) { }

    //! Accept the optional charge values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    HOSTDEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy
    /*! \param force Output parameter to write the computed force.
        \param pair_eng Output parameter to write the computed pair energy.
        \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the
       cutoff. \param torque_i The torque exerted on the i^th particle. \param torque_j The torque
       exerted on the j^th particle. \return True if they are evaluated or false if they are not
       because we are beyond the cutoff.
    */
    HOSTDEVICE FrictionResult evaluate(
                            Scalar& force_divr,
                            Scalar& pair_eng,
                            bool energy_shift)
        {

        Scalar cutoff = fast::sqrt(rcutsq) + delta;

        if (rsqr < cutoff*cutoff && lj1 != 0)
            {

            Scalar mag_dr = fast::sqrt(rsqr);

            Scalar rinv = Scalar(1.0) / mag_dr;
            Scalar rmdinv = Scalar(1.0) / (mag_dr - dlt - delta);
            Scalar rmd2inv = rmdinv * rmdinv;

            Scalar rmd6inv = rmd2inv * rmd2inv * rmd2inv;
            force_divr = rinv * rmdinv * rmd6inv * (Scalar(12.0) * lj1 * rmd6inv - Scalar(6.0) * lj2);

            // w/ 
            pair_eng = rmd6inv * (lj1 * rmd6inv - lj2);

            Scalar min = sigma*pow(2.0, 1. / 6.) + delta;
            Scalar min_sqr = min * min;
            bool contact = rsqr <= min_sqr;

            if (!contact)
                force_divr *= aes;
                pair_eng *= aes;

            pair_eng -= epsilon*(aes - Scalar(1.0));

            if (energy_shift)
                {
                Scalar rcut2inv = Scalar(1.0) / (fast::sqrt(rcutsq) - delta - dlt);
                Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
                if (rcutsq <= min_sqr)
                    pair_eng -= rcut6inv * (lj1 * rcut6inv - lj2);
                else
                    pair_eng -= aes * rcut6inv * (lj1 * rcut6inv - lj2);
                }

            return FrictionResult{true, contact};
            }
        else
            {
            return FrictionResult{false, false};
            }

        
        }

    // DEVICE Scalar evalPressureLRCIntegral()
    //     {
    //     return 0;
    //     }

    // DEVICE Scalar evalEnergyLRCIntegral()
    //     {
    //     return 0;
    //     }

#ifndef __HIPCC__
    //! Get the name of the potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return "flj";
        }

    std::string getShapeSpec() const
        {
        std::ostringstream shapedef;
        shapedef << "{\"type\": \"Friction LJ\"";
        return shapedef.str();
        }
#endif

    protected:
    Scalar rsqr; //!< Stored dr from the constructor
    Scalar rcutsq;   //!< Stored rcutsq from the constructor
    Scalar lj1;
    Scalar lj2;
    Scalar epsilon;
    Scalar aes;
    Scalar dlt;
    Scalar delta;
    Scalar sigma;
    public:
    Scalar mus;
    Scalar mur;
    Scalar ks;
    Scalar kr;
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_PAIR_FRICTION_LJ__
