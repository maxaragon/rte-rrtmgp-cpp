//
// cuda version aerosol optics, see src/Aerosol_optics.cpp by Mirjam Tijhuis
//

#include <limits>
#include <stdexcept>
#include "Aerosol_optics_rt.h"

namespace
{
    __device__
    int find_rh_class(Float rh, const Float* rh_classes)
    {
        int ihum = 0;
        Float rh_class = rh_classes[ihum];
        while (rh_class < rh)
        {
            ihum += 1;
            rh_class = rh_classes[ihum];
        }

        return ihum;
    }

    __device__
    void add_species_optics(const Float mmr, const Float dpg,
                            const Float mext, const Float ssa, const Float g,
                            Float& tau, Float& taussa, Float& taussag)
    {
        Float local_od = mmr * dpg * mext;
        tau += local_od;
        taussa += local_od * ssa;
        taussag += local_od * ssa * g;
    }


    // Accumulate species contributions to the DHG parameter buffers too.
    // Each aerosol species contributes its (g1, g2, f) weighted by its
    // scattering optical depth; the per-band caller divides at the end.
    __device__
    void add_species_optics_dhg(const Float mmr, const Float dpg,
                                const Float mext, const Float ssa, const Float g,
                                const Float g1, const Float g2, const Float f,
                                Float& tau, Float& taussa, Float& taussag,
                                Float& tau_sca_sum,
                                Float& tau_sca_g1, Float& tau_sca_g2, Float& tau_sca_f)
    {
        const Float local_od = mmr * dpg * mext;
        const Float local_sca = local_od * ssa;
        tau += local_od;
        taussa += local_sca;
        taussag += local_sca * g;
        tau_sca_sum += local_sca;
        tau_sca_g1  += local_sca * g1;
        tau_sca_g2  += local_sca * g2;
        tau_sca_f   += local_sca * f;
    }

    __global__
    void compute_from_table_kernel(
        const int ncol, const int nlay, const int ibnd, const int nbnd, const int nhum,
        const Float* aermr01, const Float* aermr02, const Float* aermr03,
        const Float* aermr04, const Float* aermr05, const Float* aermr06,
        const Float* aermr07, const Float* aermr08, const Float* aermr09,
        const Float* aermr10, const Float* aermr11,
        const Float* rh, const Float* plev, const Float* rh_classes,
        const Float* mext_phobic, const Float* ssa_phobic, const Float* g_phobic,
        const Float* mext_philic, const Float* ssa_philic, const Float* g_philic,
        Float* tau, Float* taussa, Float* taussag)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

        if ( ( icol < ncol) && ( ilay < nlay) )
        {
            int species_idx;
            Float mmr;
            Float mext;
            Float ssa;
            Float g;

            const int idx = icol + ilay * ncol;
            const int ihum = find_rh_class(rh[idx], rh_classes);

            const Float dpg = abs(plev[idx] - plev[idx + ncol]) / Float(9.81);

            // set to zero
            tau[idx] = Float(0.);
            taussa[idx] = Float(0.);
            taussag[idx] = Float(0.);

            // DU1
            species_idx = ibnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr04[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // DU2
            species_idx = ibnd + 7*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr05[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // DU3
            species_idx = ibnd + 5*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr06[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // BC1
            species_idx = ibnd + 10*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr09[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // BC2
            species_idx = ibnd + 10*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr10[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // SS1
            species_idx = ibnd + ihum*nbnd;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr01[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // SS2
            species_idx = ibnd + ihum*nbnd + 1*nbnd*nhum;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr02[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // SS3
            species_idx = ibnd + ihum*nbnd + 2*nbnd*nhum;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr03[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // SU
            species_idx = ibnd + ihum*nbnd + 4*nbnd*nhum;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr11[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // OM1
            species_idx = ibnd + 9*nbnd;
            mext = mext_phobic[species_idx];
            ssa = ssa_phobic[species_idx];
            g = g_phobic[species_idx];
            mmr = aermr08[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);

            // OM2
            species_idx = ibnd + ihum*nbnd + 3*nbnd*nhum;
            mext = mext_philic[species_idx];
            ssa = ssa_philic[species_idx];
            g = g_philic[species_idx];
            mmr = aermr07[idx];
            add_species_optics(mmr, dpg, mext, ssa, g, tau[idx], taussa[idx], taussag[idx]);
        }
    }

    __global__
    void combine_and_store_kernel(const int ncol, const int nlay, const Float tmin,
                  Float* __restrict__ tau, Float* __restrict__ ssa, Float* __restrict__ g,
                  const Float* __restrict__ ltau, const Float* __restrict__ ltaussa, const Float* __restrict__ ltaussag)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;

        if ( (icol < ncol) && (ilay < nlay) )
        {
            const int idx = icol + ilay*ncol;
            tau[idx] = ltau[idx];
            ssa[idx] = ltaussa[idx] / max(tau[idx], tmin);
            g[idx]   = ltaussag[idx] / max(ltaussa[idx], tmin);
        }
    }

    // DHG-augmented kernel: also mixes (g1, g2, f) per species by
    // scattering optical depth. Falls back to Albers 2020 hardcoded
    // values when no species contributes any scattering in the cell.
    __global__
    void compute_from_table_kernel_dhg(
        const int ncol, const int nlay, const int ibnd, const int nbnd, const int nhum,
        const Float* aermr01, const Float* aermr02, const Float* aermr03,
        const Float* aermr04, const Float* aermr05, const Float* aermr06,
        const Float* aermr07, const Float* aermr08, const Float* aermr09,
        const Float* aermr10, const Float* aermr11,
        const Float* rh, const Float* plev, const Float* rh_classes,
        const Float* mext_phobic, const Float* ssa_phobic, const Float* g_phobic,
        const Float* g1_phobic,   const Float* g2_phobic,  const Float* f_phobic,
        const Float* mext_philic, const Float* ssa_philic, const Float* g_philic,
        const Float* g1_philic,   const Float* g2_philic,  const Float* f_philic,
        Float* tau, Float* taussa, Float* taussag,
        Float* tau_sca_out, Float* tau_sca_g1, Float* tau_sca_g2, Float* tau_sca_f)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        if ( !((icol < ncol) && (ilay < nlay)) ) return;

        const int idx = icol + ilay * ncol;
        const int ihum = find_rh_class(rh[idx], rh_classes);
        const Float dpg = abs(plev[idx] - plev[idx + ncol]) / Float(9.81);

        tau[idx]     = Float(0.);
        taussa[idx]  = Float(0.);
        taussag[idx] = Float(0.);
        Float tau_sca_sum = Float(0.);
        Float tau_sca_g1_loc = Float(0.);
        Float tau_sca_g2_loc = Float(0.);
        Float tau_sca_f_loc  = Float(0.);

        #define ADD_PHOBIC(species_idx, mmr)                                            \
            add_species_optics_dhg((mmr), dpg,                                          \
                mext_phobic[species_idx], ssa_phobic[species_idx], g_phobic[species_idx],\
                g1_phobic[species_idx],   g2_phobic[species_idx],  f_phobic[species_idx],\
                tau[idx], taussa[idx], taussag[idx],                                    \
                tau_sca_sum, tau_sca_g1_loc, tau_sca_g2_loc, tau_sca_f_loc)

        #define ADD_PHILIC(species_idx, mmr)                                            \
            add_species_optics_dhg((mmr), dpg,                                          \
                mext_philic[species_idx], ssa_philic[species_idx], g_philic[species_idx],\
                g1_philic[species_idx],   g2_philic[species_idx],  f_philic[species_idx],\
                tau[idx], taussa[idx], taussag[idx],                                    \
                tau_sca_sum, tau_sca_g1_loc, tau_sca_g2_loc, tau_sca_f_loc)

        ADD_PHOBIC(ibnd,               aermr04[idx]);                        // DU1 (Woodward)
        ADD_PHOBIC(ibnd + 7*nbnd,      aermr05[idx]);                        // DU2
        ADD_PHOBIC(ibnd + 5*nbnd,      aermr06[idx]);                        // DU3
        ADD_PHOBIC(ibnd + 10*nbnd,     aermr09[idx]);                        // BC1
        ADD_PHOBIC(ibnd + 10*nbnd,     aermr10[idx]);                        // BC2
        ADD_PHILIC(ibnd + ihum*nbnd,                 aermr01[idx]);          // SS1
        ADD_PHILIC(ibnd + ihum*nbnd + 1*nbnd*nhum,   aermr02[idx]);          // SS2
        ADD_PHILIC(ibnd + ihum*nbnd + 2*nbnd*nhum,   aermr03[idx]);          // SS3
        ADD_PHILIC(ibnd + ihum*nbnd + 4*nbnd*nhum,   aermr11[idx]);          // SU
        ADD_PHOBIC(ibnd + 9*nbnd,                    aermr08[idx]);          // OM phobic
        ADD_PHILIC(ibnd + ihum*nbnd + 3*nbnd*nhum,   aermr07[idx]);          // OM philic

        #undef ADD_PHOBIC
        #undef ADD_PHILIC

        tau_sca_out[idx] = tau_sca_sum;
        tau_sca_g1 [idx] = tau_sca_g1_loc;
        tau_sca_g2 [idx] = tau_sca_g2_loc;
        tau_sca_f  [idx] = tau_sca_f_loc;
    }

    __global__
    void combine_dhg_kernel(const int ncol, const int nlay, const Float tmin,
                            Float* __restrict__ g1, Float* __restrict__ g2, Float* __restrict__ f,
                            const Float* __restrict__ tau_sca_sum,
                            const Float* __restrict__ tau_sca_g1,
                            const Float* __restrict__ tau_sca_g2,
                            const Float* __restrict__ tau_sca_f)
    {
        const int icol = blockIdx.x*blockDim.x + threadIdx.x;
        const int ilay = blockIdx.y*blockDim.y + threadIdx.y;
        if ( !((icol < ncol) && (ilay < nlay)) ) return;
        const int idx = icol + ilay*ncol;
        const Float denom = max(tau_sca_sum[idx], tmin);
        // Albers 2020 fallback when no aerosol scattering is present in
        // this cell - matches the hardcoded defaults used when DHG is off.
        if (tau_sca_sum[idx] < tmin)
        {
            g1[idx] = Float(0.962);
            g2[idx] = Float(0.50);
            f [idx] = Float(0.06);
            return;
        }
        g1[idx] = tau_sca_g1[idx] / denom;
        g2[idx] = tau_sca_g2[idx] / denom;
        f [idx] = tau_sca_f [idx] / denom;
    }

    void fill_aerosols_3d(const int ncol, const int nlay, Aerosol_concs_gpu& aerosol_concs)
    {
        for (int i=1; i<=11; ++i)
        {
            std::string name = i<10 ? "aermr0"+std::to_string(i) : "aermr"+std::to_string(i);
            if (aerosol_concs.get_vmr(name).dim(1) == 1)
            {
                aerosol_concs.set_vmr(name, aerosol_concs.get_vmr(name).subset({ {{1, ncol}, {1, nlay}}} ));
            }
        }
    }
}

Aerosol_optics_rt::Aerosol_optics_rt(
        const Array<Float,2>& band_lims_wvn, const Array<Float,1>& rh_upper,
        const Array<Float,2>& mext_phobic, const Array<Float,2>& ssa_phobic, const Array<Float,2>& g_phobic,
        const Array<Float,3>& mext_philic, const Array<Float,3>& ssa_philic, const Array<Float,3>& g_philic) :
        Optical_props_rt(band_lims_wvn)
{
    // Load coefficients.
    this->mext_phobic = mext_phobic;
    this->ssa_phobic = ssa_phobic;
    this->g_phobic = g_phobic;

    this->mext_philic = mext_philic;
    this->ssa_philic = ssa_philic;
    this->g_philic = g_philic;

    this->rh_upper = rh_upper;

    // copy to gpu.
    this->mext_phobic_gpu = this->mext_phobic;
    this->ssa_phobic_gpu = this->ssa_phobic;
    this->g_phobic_gpu = this->g_phobic;

    this->mext_philic_gpu = this->mext_philic;
    this->ssa_philic_gpu = this->ssa_philic;
    this->g_philic_gpu = this->g_philic;

    this->rh_upper_gpu = this->rh_upper;
}



void Aerosol_optics_rt::aerosol_optics(
        const int ibnd,
        Aerosol_concs_gpu& aerosol_concs,
        const Array_gpu<Float,2>& rh, const Array_gpu<Float,2>& plev,
        Optical_props_2str_rt& optical_props)
{
    const int ncol = rh.dim(1);
    const int nlay = rh.dim(2);
    const int nbnd = this->get_nband();
    const int nhum = this->rh_upper.dim(1);

    fill_aerosols_3d(ncol, nlay, aerosol_concs);

    // Temporary arrays for storage.
    Array_gpu<Float,2> ltau    ({ncol, nlay});
    Array_gpu<Float,2> ltaussa ({ncol, nlay});
    Array_gpu<Float,2> ltaussag({ncol, nlay});

    const int block_col = 64;
    const int block_lay = 1;

    const int grid_col  = ncol/block_col + (ncol%block_col > 0);
    const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

    dim3 grid_gpu(grid_col, grid_lay);
    dim3 block_gpu(block_col, block_lay);

    constexpr Float eps = std::numeric_limits<Float>::epsilon();

    compute_from_table_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, ibnd-1, nbnd, nhum,
            aerosol_concs.get_vmr("aermr01").ptr(),
            aerosol_concs.get_vmr("aermr02").ptr(),
            aerosol_concs.get_vmr("aermr03").ptr(),
            aerosol_concs.get_vmr("aermr04").ptr(),
            aerosol_concs.get_vmr("aermr05").ptr(),
            aerosol_concs.get_vmr("aermr06").ptr(),
            aerosol_concs.get_vmr("aermr07").ptr(),
            aerosol_concs.get_vmr("aermr08").ptr(),
            aerosol_concs.get_vmr("aermr09").ptr(),
            aerosol_concs.get_vmr("aermr10").ptr(),
            aerosol_concs.get_vmr("aermr11").ptr(),
            rh.ptr(), plev.ptr(),
            this->rh_upper_gpu.ptr(),
            this->mext_phobic_gpu.ptr(), this->ssa_phobic_gpu.ptr(), this->g_phobic_gpu.ptr(),
            this->mext_philic_gpu.ptr(), this->ssa_philic_gpu.ptr(), this->g_philic_gpu.ptr(),
            ltau.ptr(), ltaussa.ptr(), ltaussag.ptr());

    combine_and_store_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, eps,
            optical_props.get_tau().ptr(), optical_props.get_ssa().ptr(), optical_props.get_g().ptr(),
            ltau.ptr(), ltaussa.ptr(), ltaussag.ptr());
}


void Aerosol_optics_rt::set_dhg_tables(
        const Array<Float,2>& g1_phobic, const Array<Float,2>& g2_phobic, const Array<Float,2>& f_phobic,
        const Array<Float,3>& g1_philic, const Array<Float,3>& g2_philic, const Array<Float,3>& f_philic)
{
    this->g1_phobic_gpu = g1_phobic;
    this->g2_phobic_gpu = g2_phobic;
    this->f_phobic_gpu  = f_phobic;
    this->g1_philic_gpu = g1_philic;
    this->g2_philic_gpu = g2_philic;
    this->f_philic_gpu  = f_philic;
    this->dhg_enabled = true;
}


void Aerosol_optics_rt::aerosol_optics_dhg(
        const int ibnd,
        Aerosol_concs_gpu& aerosol_concs,
        const Array_gpu<Float,2>& rh, const Array_gpu<Float,2>& plev,
        Optical_props_2str_rt& optical_props,
        Array_gpu<Float,2>& g1_out,
        Array_gpu<Float,2>& g2_out,
        Array_gpu<Float,2>& f_out)
{
    if (!this->dhg_enabled)
        throw std::runtime_error("aerosol_optics_dhg() called before set_dhg_tables()");

    const int ncol = rh.dim(1);
    const int nlay = rh.dim(2);
    const int nbnd = this->get_nband();
    const int nhum = this->rh_upper.dim(1);

    fill_aerosols_3d(ncol, nlay, aerosol_concs);

    Array_gpu<Float,2> ltau        ({ncol, nlay});
    Array_gpu<Float,2> ltaussa     ({ncol, nlay});
    Array_gpu<Float,2> ltaussag    ({ncol, nlay});
    Array_gpu<Float,2> tau_sca_sum ({ncol, nlay});
    Array_gpu<Float,2> tau_sca_g1  ({ncol, nlay});
    Array_gpu<Float,2> tau_sca_g2  ({ncol, nlay});
    Array_gpu<Float,2> tau_sca_f   ({ncol, nlay});

    const int block_col = 64;
    const int block_lay = 1;
    const int grid_col  = ncol/block_col + (ncol%block_col > 0);
    const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);
    dim3 grid_gpu(grid_col, grid_lay);
    dim3 block_gpu(block_col, block_lay);

    constexpr Float eps = std::numeric_limits<Float>::epsilon();

    compute_from_table_kernel_dhg<<<grid_gpu, block_gpu>>>(
            ncol, nlay, ibnd-1, nbnd, nhum,
            aerosol_concs.get_vmr("aermr01").ptr(),
            aerosol_concs.get_vmr("aermr02").ptr(),
            aerosol_concs.get_vmr("aermr03").ptr(),
            aerosol_concs.get_vmr("aermr04").ptr(),
            aerosol_concs.get_vmr("aermr05").ptr(),
            aerosol_concs.get_vmr("aermr06").ptr(),
            aerosol_concs.get_vmr("aermr07").ptr(),
            aerosol_concs.get_vmr("aermr08").ptr(),
            aerosol_concs.get_vmr("aermr09").ptr(),
            aerosol_concs.get_vmr("aermr10").ptr(),
            aerosol_concs.get_vmr("aermr11").ptr(),
            rh.ptr(), plev.ptr(),
            this->rh_upper_gpu.ptr(),
            this->mext_phobic_gpu.ptr(), this->ssa_phobic_gpu.ptr(), this->g_phobic_gpu.ptr(),
            this->g1_phobic_gpu.ptr(),   this->g2_phobic_gpu.ptr(),  this->f_phobic_gpu.ptr(),
            this->mext_philic_gpu.ptr(), this->ssa_philic_gpu.ptr(), this->g_philic_gpu.ptr(),
            this->g1_philic_gpu.ptr(),   this->g2_philic_gpu.ptr(),  this->f_philic_gpu.ptr(),
            ltau.ptr(), ltaussa.ptr(), ltaussag.ptr(),
            tau_sca_sum.ptr(), tau_sca_g1.ptr(), tau_sca_g2.ptr(), tau_sca_f.ptr());

    combine_and_store_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, eps,
            optical_props.get_tau().ptr(), optical_props.get_ssa().ptr(), optical_props.get_g().ptr(),
            ltau.ptr(), ltaussa.ptr(), ltaussag.ptr());

    combine_dhg_kernel<<<grid_gpu, block_gpu>>>(
            ncol, nlay, eps,
            g1_out.ptr(), g2_out.ptr(), f_out.ptr(),
            tau_sca_sum.ptr(), tau_sca_g1.ptr(), tau_sca_g2.ptr(), tau_sca_f.ptr());
}

