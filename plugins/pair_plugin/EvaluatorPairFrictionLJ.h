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

#define HOOMD_GB_MIN(i, j) ((i > j) ? j : i)
#define HOOMD_GB_MAX(i, j) ((i > j) ? i : j)

#include "hoomd/VectorMath.h"

/*! \file EvaluatorPairFrictionLJ.h
    \brief Defines a an evaluator class for the Gay-Berne potential
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
 * Gay-Berne potential as formulated by Allen and Germano,
 * with shape-independent energy parameter, for identical uniaxial particles.
 */

class EvaluatorPairFrictionLJ
    {
    public:
    struct param_type
        {
        Scalar epsilon; //! The energy scale.
        Scalar sigma;   //! Interaction length scale
        Scalar lj1;
        Scalar lj2;
        Scalar dlt;
        Scalar mus;     //! Static friction coefficient
        Scalar muk;     //! Kinetic friction coefficient

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
            sigma = 0;
            lj1 = 0;
            lj2 = 0;
            dlt = 0;
            mus = 0;
            muk = 0;
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
            lj1 = 4.0 * epsilon * pow(dsigma, 12.0);
            lj2 = 4.0 * epsilon * pow(dsigma, 6.0);
            dlt = delta;
            mus = v["mus"].cast<Scalar>();
            muk = v["muk"].cast<Scalar>();
            }

        pybind11::dict toPython()
            {
            pybind11::dict v;
            v["epsilon"] = epsilon;
            v["sigma"] = sigma;
            v["delta"] = dlt;
            v["mus"] = mus;
            v["muk"] = muk;
            return v;
            }

#endif
        }
#ifdef SINGLE_PRECISION
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    // Nullary structure required by AnisoPotentialPair.
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
    HOSTDEVICE EvaluatorPairFrictionLJ(const Scalar3& _dr,
                               const Scalar4& _qi,
                               const Scalar4& _qj,
                               const Scalar _rcutsq,
                               const param_type& _params)
        : dr(_dr), rcutsq(_rcutsq), qi(_qi), qj(_qj), lj1(_params.lj1),
          lj2(_params.lj2), dlt(dlt), mus(_params.mus), muk(_params.muk)
        {
        }

    //! uses diameter
    HOSTDEVICE static bool needsDiameter()
        {
        return false;
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
    HOSTDEVICE void setDiameter(Scalar di, Scalar dj) { }

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
    HOSTDEVICE bool evaluate(Scalar3& force,
                             Scalar& pair_eng,
                             bool energy_shift,
                             Scalar3& torque_i,
                             Scalar3& torque_j)
        {

        Scalar rsq = dot(dr, dr);

        if (rsq < rcutsq && lj1 != 0)
            {

            Scalar mag_dr = fast::sqrt(rsq);

            Scalar rinv = Scalar(1.0) / (mag_dr - dlt);
            Scalar r2inv = rinv * rinv;

            Scalar r6inv = r2inv * r2inv * r2inv;
            Scalar force_divr = r2inv * r6inv * (Scalar(12.0) * lj1 * r6inv - Scalar(6.0) * lj2);

            // auto mag_force = force_divr * mag_dr;

            pair_eng = r6inv * (lj1 * r6inv - lj2);
        
            vec3<Scalar> unitr = fast::rsqrt(dot(dr, dr)) * dr;

            // obtain rotation matrices (space->body)
            rotmat3<Scalar> rotA(conj(qi));
            rotmat3<Scalar> rotB(conj(qj));

            // last row of rotation matrix
            vec3<Scalar> a3 = rotA.row2;
            vec3<Scalar> b3 = rotB.row2;

            Scalar ca = dot(a3, unitr);
            Scalar cb = dot(b3, unitr);
            Scalar cab = dot(a3, b3);

            // compute vector force and torque
            vec3<Scalar> f = force_divr * dr;
            force = vec_to_scalar3(f);

            vec3<Scalar> rca = Scalar(1.0 / 2.0)
                               * (-dr - r * chi_fact * ((ca - chic * cb) * a3 - (cb - chic * ca) * b3));
            vec3<Scalar> rcb = rca + dr;
            torque_i = vec_to_scalar3(cross(rca, fK));
            torque_j = -vec_to_scalar3(cross(rcb, fK));

            if (energy_shift)
                {
                Scalar rcut2inv = Scalar(1.0) / rcutsq;
                Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
                pair_eng -= rcut6inv * (lj1 * rcut6inv - lj2);
                }

            return true;
            }
        else
            {
            return false;
            }

        
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
    vec3<Scalar> dr; //!< Stored dr from the constructor
    Scalar rcutsq;   //!< Stored rcutsq from the constructor
    quat<Scalar> qi; //!< Orientation quaternion for particle i
    quat<Scalar> qj; //!< Orientation quaternion for particle j
    Scalar lj1;
    Scalar lj2;
    Scalar dlt;
    Scalar mus;
    Scalar muk;
    // const param_type &params;  //!< The pair potential parameters
    };

    } // end namespace md
    } // end namespace hoomd

#undef HOOMD_GB_MIN
#undef HOOMD_GB_MAX
#endif // __EVALUATOR_PAIR_FRICTION_LJ__
