"""Band-average monochromatic DHG parameters onto the RRTMGP-SW 14 bands.

Inputs (this folder):
  aerosol_optics_dhg.nc                         (from fit_dhg.py)
  *_optics_IFS_2023-06-19.nc                    (Bozzo monochromatic LUTs)
  aerosol_optics_sw_source.nc                   (menno's per-band LUT, for
                                                 band edges + fallback data)

Output:
  aerosol_optics_sw_dhg.nc                      (drop-in extension of the
                                                 menno schema with new
                                                 g1/g2/f_{hydrophilic,hydrophobic})

Band average weight:  w(lambda) = ext_mono(lambda) * ssa_mono(lambda)
(i.e. scattered-flux weight, TSI treated as flat within a band).

Species-slot mapping (see plan file for rationale). Slots that Bozzo's
five family files do not cover fall back to single-HG: g1 = g2 =
asymmetry, f = 1 -- the kernel output is then unchanged for those
slots, which is the honest choice when no DHG data exist.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import netCDF4 as nc

ROOT = Path(__file__).resolve().parent

SRC_LUT       = ROOT / "aerosol_optics_sw_source.nc"
DHG_MONO      = ROOT / "aerosol_optics_dhg.nc"
OUT_LUT       = ROOT / "aerosol_optics_sw_dhg.nc"

BOZZO_FILES = {
    "sea_salt":     "sea_salt_optics_IFS_2023-06-19.nc",
    "dust":         "dust_Woodward_optics_IFS_2023-06-19.nc",
    "organic":      "organic_optics_IFS_2023-06-19.nc",
    "black_carbon": "black_carbon_Hess_optics_IFS_2023-06-19.nc",
    "sulfate":      "sulfate_optics_IFS_2023-06-19.nc",
}

# Mapping: target species slot -> Bozzo source, or None -> single-HG fallback.
# Each entry is (family_name, size_bin, rh_override).
#   rh_override=None      => use all 12 RH bins from Bozzo directly
#   rh_override=float(%)  => use that RH bin only (replicated as needed)
HYDROPHILIC_MAP = [
    ("sea_salt", 0, None),   # slot 0  SS1
    ("sea_salt", 1, None),   # slot 1  SS2
    ("sea_salt", 2, None),   # slot 2  SS3
    ("organic",  0, None),   # slot 3  OM
    ("sulfate",  0, None),   # slot 4  SU
    (None,       0, None),   # slot 5  OB (SOA-biogenic)       -> fallback
    (None,       0, None),   # slot 6  OA (SOA-anthropogenic)  -> fallback
]

HYDROPHOBIC_MAP = [
    (None,       0, None),   # slot 0  DD Dubovik bin1
    (None,       1, None),   # slot 1  DD Dubovik bin2
    (None,       2, None),   # slot 2  DD Dubovik bin3
    (None,       0, None),   # slot 3  DD Fouquart bin1
    (None,       1, None),   # slot 4  DD Fouquart bin2
    (None,       2, None),   # slot 5  DD Fouquart bin3
    ("dust",     0, 0.0),    # slot 6  DD Woodward bin1  <- Bozzo
    ("dust",     1, 0.0),    # slot 7  DD Woodward bin2  <- Bozzo
    ("dust",     2, 0.0),    # slot 8  DD Woodward bin3  <- Bozzo
    ("organic",  0, 0.0),    # slot 9  OM phobic         <- Bozzo RH=0 slice
    ("black_carbon", 0, 0.0),# slot 10 BC OPAC/Hess      <- Bozzo
    (None,           0, None),# slot 11 BC Bond&Bergstrom -> fallback
    (None,           0, None),# slot 12 BC Stier          -> fallback
    ("sulfate",  0, 0.0),    # slot 13 strat sulfate      <- Bozzo RH=0 slice
]


def _canonicalise_dhg(g1, g2, f):
    """Enforce f >= 0.5 so g1 always labels the dominant lobe. Prevents
    linear parameter averaging from being destroyed by label flips between
    neighbouring wavelengths."""
    swap = f < 0.5
    g1_new = np.where(swap, g2, g1)
    g2_new = np.where(swap, g1, g2)
    f_new  = np.where(swap, 1.0 - f, f)
    return g1_new, g2_new, f_new


def load_bozzo_all():
    """Load monochromatic ext/ssa and DHG (g1,g2,f) per Bozzo family."""
    data = {}
    with nc.Dataset(DHG_MONO) as dhg_root:
        for name, fn in BOZZO_FILES.items():
            with nc.Dataset(ROOT / fn) as f:
                wave = np.asarray(f.variables["wavelength"][:], dtype=np.float64)
                rh   = np.asarray(f.variables["rel_hum"][:], dtype=np.float64)
                ext  = np.asarray(f.variables["extinction"][:], dtype=np.float64)
                ssa  = np.asarray(f.variables["single_scatter_albedo"][:], dtype=np.float64)
                asy  = np.asarray(f.variables["asymmetry_factor"][:], dtype=np.float64)
            grp = dhg_root.groups[name]
            g1 = np.asarray(grp.variables["g1"][:], dtype=np.float64)
            g2 = np.asarray(grp.variables["g2"][:], dtype=np.float64)
            ff = np.asarray(grp.variables["f"][:], dtype=np.float64)
            g1, g2, ff = _canonicalise_dhg(g1, g2, ff)
            data[name] = dict(
                wavelength=wave, rel_hum=rh,
                ext=ext, ssa=ssa, asymmetry=asy,
                g1=g1, g2=g2, f=ff,
            )
    return data


def _band_average_slice(wave_mono, ext_mono, ssa_mono,
                        g1_mono, g2_mono, f_mono,
                        lam_lo, lam_hi):
    """Average (g1,g2,f) over the band [lam_lo, lam_hi] weighted by ext*ssa."""
    mask = (wave_mono >= lam_lo) & (wave_mono <= lam_hi)
    status = "bozzo-fit"
    if mask.sum() == 0:
        # No in-band sample: evaluate at band midpoint via linear interp in wavelength.
        lam_mid = 0.5 * (lam_lo + lam_hi)
        if lam_mid < wave_mono.min() or lam_mid > wave_mono.max():
            status = "extrapolated"
        # np.interp clamps to end values outside the range, which is the
        # nearest-neighbour behaviour we want.
        g1 = float(np.interp(lam_mid, wave_mono, g1_mono))
        g2 = float(np.interp(lam_mid, wave_mono, g2_mono))
        ff = float(np.interp(lam_mid, wave_mono, f_mono))
        return g1, g2, ff, status
    w = ext_mono[mask] * ssa_mono[mask]
    denom = w.sum()
    if denom <= 0.0:
        # Degenerate (perfectly absorbing across band) -> fall back to arithmetic mean.
        return (float(g1_mono[mask].mean()),
                float(g2_mono[mask].mean()),
                float(f_mono [mask].mean()),
                "unweighted")
    g1 = float((w * g1_mono[mask]).sum() / denom)
    g2 = float((w * g2_mono[mask]).sum() / denom)
    ff = float((w * f_mono [mask]).sum() / denom)
    return g1, g2, ff, status


def _band_average_scalar(wave_mono, ext_mono, ssa_mono, y_mono,
                         lam_lo, lam_hi):
    """Band-average scalar y(lambda) with the same ext*ssa weight."""
    mask = (wave_mono >= lam_lo) & (wave_mono <= lam_hi)
    if mask.sum() == 0:
        lam_mid = 0.5 * (lam_lo + lam_hi)
        return float(np.interp(lam_mid, wave_mono, y_mono))
    w = ext_mono[mask] * ssa_mono[mask]
    denom = w.sum()
    if denom <= 0.0:
        return float(y_mono[mask].mean())
    return float((w * y_mono[mask]).sum() / denom)


def band_average_species(bozzo, family, size_bin, rh_override,
                         wn_lo, wn_hi, n_rh_target,
                         asy_fallback):
    """Return g1/g2/f arrays shaped (n_rh_target, nband) plus source flag.

    If rh_override is None: map Bozzo's 12 RH bins 1:1 onto n_rh_target=12.
    If rh_override is a float %: use that RH slice for every output RH bin
    (or the single RH bin, for the RH=1 case on the output).
    """
    src = bozzo[family]
    wave = src["wavelength"]            # m
    rh_src = src["rel_hum"]             # %
    ext = src["ext"]                    # (nw, nrh, nsize)
    ssa = src["ssa"]                    # (nw, nrh, nsize)
    g1  = src["g1"]
    g2  = src["g2"]
    ff  = src["f"]

    nband = wn_lo.size
    # Convert wavenumber edges (cm^-1) to wavelengths (m); swap high/low.
    lam_lo = 1.0 / (wn_hi * 100.0)
    lam_hi = 1.0 / (wn_lo * 100.0)

    out_g1 = np.full((n_rh_target, nband), np.nan, dtype=np.float32)
    out_g2 = np.full_like(out_g1, np.nan)
    out_f  = np.full_like(out_g1, np.nan)
    out_asy_self = np.full_like(out_g1, np.nan)  # self-weighted asy, for QC
    out_status = np.empty((n_rh_target, nband), dtype=object)

    asy_mono = src["asymmetry"]                   # (nw, nrh, nsize)

    for ir_out in range(n_rh_target):
        if rh_override is None:
            ir_src = ir_out
        else:
            # Nearest RH bin in the source to the requested percentage.
            ir_src = int(np.argmin(np.abs(rh_src - rh_override)))

        ext_col = ext[:, ir_src, size_bin]
        ssa_col = ssa[:, ir_src, size_bin]
        asy_col = asy_mono[:, ir_src, size_bin]
        g1_col  = g1 [:, ir_src, size_bin]
        g2_col  = g2 [:, ir_src, size_bin]
        f_col   = ff [:, ir_src, size_bin]

        for ib in range(nband):
            g1b, g2b, fb, status = _band_average_slice(
                wave, ext_col, ssa_col, g1_col, g2_col, f_col,
                lam_lo[ib], lam_hi[ib],
            )
            asy_b_self = _band_average_scalar(
                wave, ext_col, ssa_col, asy_col, lam_lo[ib], lam_hi[ib],
            )
            if status == "extrapolated":
                # Outside Bozzo's wavelength coverage: use the tabulated
                # band-averaged asymmetry as a single-HG collapse. Honest:
                # no DHG information is available for this band.
                asy_b = float(asy_fallback[ir_out, ib]
                              if asy_fallback.ndim == 2
                              else asy_fallback[ib])
                g1b, g2b, fb = asy_b, asy_b, 1.0
                asy_b_self = asy_b
            out_g1[ir_out, ib] = g1b
            out_g2[ir_out, ib] = g2b
            out_f [ir_out, ib] = fb
            out_asy_self[ir_out, ib] = asy_b_self
            out_status[ir_out, ib] = status

    return out_g1, out_g2, out_f, out_asy_self, out_status


def _copy_structure(src, dst):
    """Copy every dimension, variable and attribute from src to dst."""
    for attr in src.ncattrs():
        dst.setncattr(attr, src.getncattr(attr))
    for dname, dim in src.dimensions.items():
        dst.createDimension(dname, len(dim))
    for vname, var in src.variables.items():
        newv = dst.createVariable(vname, var.dtype, var.dimensions,
                                  fill_value=getattr(var, "_FillValue", None)
                                  if "_FillValue" in var.ncattrs() else None)
        for a in var.ncattrs():
            if a == "_FillValue":
                continue
            newv.setncattr(a, var.getncattr(a))
        newv[:] = var[:]


def main():
    print(f"[load] Bozzo monochromatic ext/ssa + DHG ...")
    bozzo = load_bozzo_all()

    print(f"[load] target schema from {SRC_LUT}")
    with nc.Dataset(SRC_LUT) as src:
        wn_lo = np.asarray(src.variables["wavenumber1"][:], dtype=np.float64)
        wn_hi = np.asarray(src.variables["wavenumber2"][:], dtype=np.float64)
        nband = wn_lo.size
        n_rh  = len(src.dimensions["relative_humidity"])
        asy_phil = np.asarray(src.variables["asymmetry_hydrophilic"][:],
                              dtype=np.float32)  # (nhyd_phil, nrh, nband)
        asy_phob = np.asarray(src.variables["asymmetry_hydrophobic"][:],
                              dtype=np.float32)  # (nhyd_phob, nband)
        nhyd_phil = asy_phil.shape[0]
        nhyd_phob = asy_phob.shape[0]
    assert n_rh == 12 and nband == 14
    assert nhyd_phil == len(HYDROPHILIC_MAP)
    assert nhyd_phob == len(HYDROPHOBIC_MAP)

    # Build the DHG arrays for each slot.
    out_g1_phil = np.empty((nhyd_phil, n_rh, nband), dtype=np.float32)
    out_g2_phil = np.empty_like(out_g1_phil)
    out_f_phil  = np.empty_like(out_g1_phil)
    src_phil    = np.empty(nhyd_phil, dtype=object)

    out_g1_phob = np.empty((nhyd_phob, nband), dtype=np.float32)
    out_g2_phob = np.empty_like(out_g1_phob)
    out_f_phob  = np.empty_like(out_g1_phob)
    src_phob    = np.empty(nhyd_phob, dtype=object)

    max_dev_phil = 0.0
    max_dev_phob = 0.0

    print(f"[map ] hydrophilic slots")
    for i, (family, size_bin, rh_ov) in enumerate(HYDROPHILIC_MAP):
        if family is None:
            # Single-HG fallback: DHG collapses to the tabulated asymmetry.
            out_g1_phil[i] = asy_phil[i]
            out_g2_phil[i] = asy_phil[i]
            out_f_phil [i] = 1.0
            src_phil[i] = "single-hg-fallback"
            print(f"  {i}: fallback (no Bozzo source)")
            continue
        g1, g2, ff, asy_self, stat = band_average_species(
            bozzo, family, size_bin, rh_ov, wn_lo, wn_hi, n_rh,
            asy_fallback=asy_phil[i],
        )
        out_g1_phil[i] = g1
        out_g2_phil[i] = g2
        out_f_phil [i] = ff
        # Self-consistency check: reconstruct asymmetry from DHG params
        # and compare against the Bozzo monochromatic asymmetry band-
        # averaged with the *same* weight.
        recon = ff * g1 + (1.0 - ff) * g2
        dev = np.abs(recon - asy_self).max()
        max_dev_phil = max(max_dev_phil, float(dev))
        # Flag any extrapolated/unweighted bands.
        flags = {s for row in stat for s in row}
        src_phil[i] = "bozzo-fit" if flags == {"bozzo-fit"} else (
            "bozzo-fit+" + "+".join(sorted(flags - {"bozzo-fit"}))
        )
        print(f"  {i}: {family}[size={size_bin}]  max|recon-asy|={dev:.3f}  "
              f"flags={src_phil[i]}")

    print(f"[map ] hydrophobic slots")
    for i, (family, size_bin, rh_ov) in enumerate(HYDROPHOBIC_MAP):
        if family is None:
            out_g1_phob[i] = asy_phob[i]
            out_g2_phob[i] = asy_phob[i]
            out_f_phob [i] = 1.0
            src_phob[i] = "single-hg-fallback"
            print(f"  {i}: fallback (no Bozzo source)")
            continue
        g1, g2, ff, asy_self, stat = band_average_species(
            bozzo, family, size_bin, rh_ov, wn_lo, wn_hi, n_rh_target=1,
            asy_fallback=asy_phob[i][None, :],
        )
        out_g1_phob[i] = g1[0]
        out_g2_phob[i] = g2[0]
        out_f_phob [i] = ff[0]
        recon = ff[0] * g1[0] + (1.0 - ff[0]) * g2[0]
        dev = np.abs(recon - asy_self[0]).max()
        max_dev_phob = max(max_dev_phob, float(dev))
        flags = {s for s in stat[0]}
        src_phob[i] = "bozzo-fit" if flags == {"bozzo-fit"} else (
            "bozzo-fit+" + "+".join(sorted(flags - {"bozzo-fit"}))
        )
        print(f"  {i}: {family}[size={size_bin}]  max|recon-asy|={dev:.3f}  "
              f"flags={src_phob[i]}")

    print(f"[qc  ] max |recon-asy|: philic={max_dev_phil:.3f}  "
          f"phobic={max_dev_phob:.3f}")
    # Threshold 0.20 accommodates the known 2-term HG expressivity limit on
    # coarse sea salt (bozzo sz2) in bands containing the rainbow/glory —
    # the fit residual is visible in the monochromatic DHG as well.
    if max(max_dev_phil, max_dev_phob) > 0.20:
        print("[qc  ] WARNING: reconstruction deviates more than 0.20 from "
              "band-averaged asymmetry for at least one slot", file=sys.stderr)
        sys.exit(1)

    # Write the output file: copy the source schema verbatim, then append
    # DHG variables.
    print(f"[emit] {OUT_LUT}")
    with nc.Dataset(SRC_LUT) as src, nc.Dataset(OUT_LUT, "w") as out:
        _copy_structure(src, out)
        out.setncattr("dhg_provenance",
                      "Double-HG (g1, g2, f) fitted to Bozzo IFS phase "
                      "functions (fit_dhg.py) then band-averaged to "
                      "RRTMGP-SW bands with weight ext(lambda) * ssa(lambda).")
        # New: source dimension for the dhg_source_* char arrays.
        out.createDimension("src_code_len", 64)

        def make(name, dims, data, long_name):
            v = out.createVariable(name, "f4", dims,
                                   fill_value=np.float32(np.nan))
            v.units = "1"
            v.long_name = long_name
            v[:] = data

        make("g1_hydrophilic",
             ("hydrophilic", "relative_humidity", "band"),
             out_g1_phil,
             "DHG forward-lobe asymmetry g1 for hydrophilic aerosols")
        make("g2_hydrophilic",
             ("hydrophilic", "relative_humidity", "band"),
             out_g2_phil,
             "DHG back-lobe asymmetry g2 for hydrophilic aerosols")
        make("f_hydrophilic",
             ("hydrophilic", "relative_humidity", "band"),
             out_f_phil,
             "DHG forward-lobe weight f for hydrophilic aerosols")
        make("g1_hydrophobic", ("hydrophobic", "band"),
             out_g1_phob,
             "DHG forward-lobe asymmetry g1 for hydrophobic aerosols")
        make("g2_hydrophobic", ("hydrophobic", "band"),
             out_g2_phob,
             "DHG back-lobe asymmetry g2 for hydrophobic aerosols")
        make("f_hydrophobic", ("hydrophobic", "band"),
             out_f_phob,
             "DHG forward-lobe weight f for hydrophobic aerosols")

        v = out.createVariable("dhg_source_hydrophilic", "S1",
                               ("hydrophilic", "src_code_len"))
        v.long_name = "provenance flag for each hydrophilic slot"
        padded = np.array([s.ljust(64)[:64] for s in src_phil], dtype="S64")
        v[:] = nc.stringtochar(padded)

        v = out.createVariable("dhg_source_hydrophobic", "S1",
                               ("hydrophobic", "src_code_len"))
        v.long_name = "provenance flag for each hydrophobic slot"
        padded = np.array([s.ljust(64)[:64] for s in src_phob], dtype="S64")
        v[:] = nc.stringtochar(padded)

    print(f"[done] wrote {OUT_LUT}")


if __name__ == "__main__":
    main()
