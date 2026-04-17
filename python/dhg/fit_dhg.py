"""Fit Double Henyey-Greenstein (g1, g2, f) to the Bozzo IFS aerosol
phase functions.

Reads every `*_optics_IFS_2023-06-19.nc` in this folder, fits DHG per
(wavelength, rel_hum, size_bin) slice using a log-space, sin(theta)-weighted
nonlinear least squares, and writes aerosol_optics_dhg.nc (one group per
species) with variables g1, g2, f, fit_cost, plus a DHG-reconstructed
asymmetry for QC.

Bozzo phase-function normalisation: angle in degrees over [0, 180];
pfun(cos(ang)) * d(cos(ang)) on [-1, 1] integrates to 2 (=> int dOmega = 4 pi).
DHG in matching convention:
    P_DHG(mu; g1, g2, f) = f (1-g1^2)/(1+g1^2 - 2 g1 mu)^(3/2)
                        + (1-f)(1-g2^2)/(1+g2^2 - 2 g2 mu)^(3/2)
integrates to 2 on mu in [-1, 1] for any (g1, g2, f) with |g_i| < 1.
"""
from pathlib import Path

import numpy as np
import netCDF4 as nc
from scipy.optimize import least_squares

SRC_DIR = Path(__file__).resolve().parent
OUT_PATH = SRC_DIR / "aerosol_optics_dhg.nc"

SPECIES_FILES = {
    "sea_salt":     "sea_salt_optics_IFS_2023-06-19.nc",
    "dust":         "dust_Woodward_optics_IFS_2023-06-19.nc",
    "organic":      "organic_optics_IFS_2023-06-19.nc",
    "black_carbon": "black_carbon_Hess_optics_IFS_2023-06-19.nc",
    "sulfate":      "sulfate_optics_IFS_2023-06-19.nc",
}


def hg(mu, g):
    g2 = g * g
    denom = np.maximum(1.0 + g2 - 2.0 * g * mu, 1e-30)
    return (1.0 - g2) / (denom * np.sqrt(denom))


def dhg(mu, g1, g2, f):
    return f * hg(mu, g1) + (1.0 - f) * hg(mu, g2)


def fit_one(theta_rad, P, g_init):
    mu = np.cos(theta_rad)
    w = np.sqrt(np.sin(theta_rad) + 1e-12)
    P_safe = np.maximum(P, 1e-300)
    logP = np.log(P_safe)

    def residual(x):
        g1, g2, f = x
        model = dhg(mu, g1, g2, f)
        return w * (np.log(np.maximum(model, 1e-300)) - logP)

    x0 = np.array([min(0.98, max(0.10, float(g_init))), 0.0, 0.8])
    res = least_squares(
        residual, x0,
        bounds=([0.0, -0.999, 0.0], [0.999, 0.999, 1.0]),
        xtol=1e-10, ftol=1e-10, max_nfev=500,
    )
    g1, g2, f = map(float, res.x)
    if g1 < g2:
        g1, g2, f = g2, g1, 1.0 - f
    return g1, g2, f, float(res.cost)


def fit_file(path):
    with nc.Dataset(path) as f:
        angle = np.asarray(f.variables["angle"][:], dtype=np.float64)
        P_all = np.asarray(f.variables["phase_function"][:], dtype=np.float64)
        g_all = np.asarray(f.variables["asymmetry_factor"][:], dtype=np.float64)
        wave = np.asarray(f.variables["wavelength"][:], dtype=np.float64)
        rh = np.asarray(f.variables["rel_hum"][:], dtype=np.float64)
    theta = np.deg2rad(angle)
    nw, nrh, nsize, _ = P_all.shape
    G1 = np.zeros((nw, nrh, nsize))
    G2 = np.zeros_like(G1)
    F  = np.zeros_like(G1)
    COST = np.zeros_like(G1)
    for iw in range(nw):
        for ir in range(nrh):
            for isz in range(nsize):
                P = P_all[iw, ir, isz]
                if not np.all(np.isfinite(P)) or P.max() <= 0.0:
                    continue
                g_init = g_all[iw, ir, isz]
                g1, g2, f_, cost = fit_one(theta, P, g_init)
                G1[iw, ir, isz]   = g1
                G2[iw, ir, isz]   = g2
                F[iw, ir, isz]    = f_
                COST[iw, ir, isz] = cost
    g_recon = F * G1 + (1.0 - F) * G2
    return dict(wave=wave, rh=rh, G1=G1, G2=G2, F=F, COST=COST,
                g_orig=g_all, g_recon=g_recon)


def main():
    records = {}
    for name, fn in SPECIES_FILES.items():
        print(f"fitting {name:14s} from {fn}")
        rec = fit_file(SRC_DIR / fn)
        records[name] = rec
        abs_err = np.abs(rec["g_recon"] - rec["g_orig"])
        print(
            f"  shape {rec['G1'].shape}  mean cost {rec['COST'].mean():.3e}"
            f"   max cost {rec['COST'].max():.3e}"
            f"   |g_recon - g_orig| max {abs_err.max():.2e}"
        )

    with nc.Dataset(OUT_PATH, "w") as out:
        out.source = "fit of DHG (f*HG(g1)+(1-f)*HG(g2)) to Bozzo IFS phase functions"
        out.convention = ("Bozzo phase_function normalised so int_{-1}^{+1} P dmu = 2; "
                          "DHG fit minimises sin(theta)-weighted log residuals")
        for name, rec in records.items():
            grp = out.createGroup(name)
            nw, nrh, nsize = rec["G1"].shape
            grp.createDimension("wavelength", nw)
            grp.createDimension("rel_hum", nrh)
            grp.createDimension("size_bin", nsize)
            v = grp.createVariable("wavelength", "f4", ("wavelength",))
            v.units = "m"
            v[:] = rec["wave"]
            v = grp.createVariable("rel_hum", "f4", ("rel_hum",))
            v.units = "%"
            v[:] = rec["rh"]
            dims3 = ("wavelength", "rel_hum", "size_bin")
            for nm, arr, ln in [
                ("g1", rec["G1"], "DHG forward-lobe asymmetry"),
                ("g2", rec["G2"], "DHG second-lobe asymmetry"),
                ("f",  rec["F"],  "DHG weight of forward lobe"),
                ("fit_cost",   rec["COST"],    "nonlinear LSQ cost"),
                ("g_original", rec["g_orig"],  "asymmetry_factor from source file"),
                ("g_recon",    rec["g_recon"], "f*g1+(1-f)*g2 reconstruction"),
            ]:
                v = grp.createVariable(nm, "f4", dims3)
                v.long_name = ln
                v[:] = arr
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
