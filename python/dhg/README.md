# DHG aerosol phase-function pipeline

Scripts that produce the Double-Henyey–Greenstein (DHG) LUT consumed by
the backward ray tracer when launched with `--aerosol-dhg`.

## Inputs (not shipped)

Five per-species IFS/CAMS aerosol optics files from Bozzo et al. 2020,
dataset DOI <https://doi.org/10.24380/jgs8-sc58>:

```
sea_salt_optics_IFS_2023-06-19.nc
dust_Woodward_optics_IFS_2023-06-19.nc
organic_optics_IFS_2023-06-19.nc
black_carbon_Hess_optics_IFS_2023-06-19.nc
sulfate_optics_IFS_2023-06-19.nc
```

Each carries `phase_function(wavelength, rel_hum, size_bin, angle=3600)`
— so the Mie work is already done; we only fit DHG to the sampled
phase function.

Also needed (for band edges + the hydrophilic/hydrophobic slot layout):
a baseline `aerosol_optics_sw.nc` in the menno schema (`band`,
`hydrophilic=7`, `hydrophobic=14`, `relative_humidity=12`).

## Pipeline (two steps)

1. **`fit_dhg.py`** — fits `(g1, g2, f)` per slice of every Bozzo file
   using a log-space, sin(θ)-weighted nonlinear least-squares. Emits
   `aerosol_optics_dhg.nc` (one group per species family, 19
   monochromatic wavelengths).
2. **`band_average_dhg.py`** — averages the monochromatic fits onto
   the 14 RRTMGP-SW bands with weight `ext(λ) · ssa(λ)`
   (TSI assumed ≈ flat within a band), maps each target species slot
   (SS1/2/3, OM, SU, OB, OA, DD variants, BC variants, strat SU) onto
   the Bozzo family/size/RH it was fitted from, and emits
   `aerosol_optics_sw_dhg.nc` in the same schema as the base aerosol
   LUT. Slots Bozzo does not cover (SOA, DD Dubovik/Fouquart, BC Bond &
   Bergstrom / Stier) fall back to single-HG
   (`g1 = g2 = asymmetry, f = 1`) — kernel output is unchanged for
   those slots.

The produced `data/aerosol_optics_sw_dhg.nc` is versioned; regenerate it
only when the Bozzo source tables change.

## Enabling at run-time

```
./test_rte_rrtmgp_bw --aerosol-optics --aerosol-dhg ...
```

Omit `--aerosol-dhg` for the legacy single-HG code path — that path is
unchanged.
