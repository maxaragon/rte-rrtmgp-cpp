#ifndef RAYTRACER_DEFINITIONS_H
#define RAYTRACER_DEFINITIONS_H

#include "types.h"

namespace Raytracer_definitions
{
    template<typename T>
    struct Vector
    {
        T x;
        T y;
        T z;
    };


    struct Optics_scat
    {
        Float k_sca_gas;
        Float k_sca_cld;
        Float k_sca_aer;
        Float asy_cld;
        Float asy_aer;
        // Add DHG parameters for aerosols
        Float asy_aer_g1;
        Float asy_aer_g2;
        Float asy_aer_f; 
        Float asy_aer_fb;  // backscatter peak term for aerosols
    };


    enum class Photon_kind { Direct, Diffuse };

    struct Photon
    {
        Vector<Float> position;
        Vector<Float> direction;
        Photon_kind kind;
    };
}
#endif
