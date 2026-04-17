//
// cuda version aerosol optics, see src/Aerosol_optics.cpp by Mirjam Tijhuis
//

#ifndef AEROSOL_OPTICS_RT_H
#define AEROSOL_OPTICS_RT_H

#include "Array.h"
#include "Optical_props_rt.h"
#include "Gas_concs.h"
#include "types.h"

using Aerosol_concs_gpu = Gas_concs_gpu;
using Aerosol_concs = Gas_concs;

// Forward declarations.
class Optical_props_rt;

#ifdef USECUDA
class Aerosol_optics_rt : public Optical_props_rt
{
    public:
        Aerosol_optics_rt(
                const Array<Float,2>& band_lims_wvn, const Array<Float,1>& rh_upper,
                const Array<Float,2>& mext_phobic, const Array<Float,2>& ssa_phobic, const Array<Float,2>& g_phobic,
                const Array<Float,3>& mext_philic, const Array<Float,3>& ssa_philic, const Array<Float,3>& g_philic);


        void aerosol_optics(
                const int ibnd,
                Aerosol_concs_gpu& aerosol_concs,
                const Array_gpu<Float,2>& rh, const Array_gpu<Float,2>& plev,
                Optical_props_2str_rt& optical_props);

        // Populate DHG (Double Henyey-Greenstein) lookup tables. Enables
        // the DHG aerosol sampling path in the backward ray tracer. Must be
        // called exactly once per Aerosol_optics_rt instance before any
        // aerosol_optics_dhg() call.
        void set_dhg_tables(
                const Array<Float,2>& g1_phobic, const Array<Float,2>& g2_phobic, const Array<Float,2>& f_phobic,
                const Array<Float,3>& g1_philic, const Array<Float,3>& g2_philic, const Array<Float,3>& f_philic);

        // Same as aerosol_optics(), but also emits per-column (g1, g2, f)
        // arrays mixed by scattering optical depth across the 11 CAMS
        // species. Requires set_dhg_tables() to have been called.
        void aerosol_optics_dhg(
                const int ibnd,
                Aerosol_concs_gpu& aerosol_concs,
                const Array_gpu<Float,2>& rh, const Array_gpu<Float,2>& plev,
                Optical_props_2str_rt& optical_props,
                Array_gpu<Float,2>& g1_out,
                Array_gpu<Float,2>& g2_out,
                Array_gpu<Float,2>& f_out);

        bool dhg_available() const { return this->dhg_enabled; }

    private:
        // Lookup table coefficients
        Array<Float,2> mext_phobic;
        Array<Float,2> ssa_phobic;
        Array<Float,2> g_phobic;

        Array<Float,3> mext_philic;
        Array<Float,3> ssa_philic;
        Array<Float,3> g_philic;

        Array<Float,1> rh_upper;

        // gpu versions
        Array_gpu<Float,2> mext_phobic_gpu;
        Array_gpu<Float,2> ssa_phobic_gpu;
        Array_gpu<Float,2> g_phobic_gpu;

        Array_gpu<Float,3> mext_philic_gpu;
        Array_gpu<Float,3> ssa_philic_gpu;
        Array_gpu<Float,3> g_philic_gpu;

        Array_gpu<Float,1> rh_upper_gpu;

        // Optional DHG (Double Henyey-Greenstein) coefficients.
        bool dhg_enabled = false;
        Array_gpu<Float,2> g1_phobic_gpu;
        Array_gpu<Float,2> g2_phobic_gpu;
        Array_gpu<Float,2> f_phobic_gpu;
        Array_gpu<Float,3> g1_philic_gpu;
        Array_gpu<Float,3> g2_philic_gpu;
        Array_gpu<Float,3> f_philic_gpu;
};
#endif

#endif
