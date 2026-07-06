# RCEMIP: Radiative-Convective Equilibrium MIP

This directory sets up the RCEMIP-I `RCE_small` configuration of
Wing et al. (2018), *Radiative-convective equilibrium model intercomparison
project*, Geosci. Model Dev. 11, 793-813, doi:10.5194/gmd-11-793-2018.

Radiative-convective equilibrium over a doubly-periodic, non-rotating,
fixed-SST ocean:

- 96 km x 96 km horizontal domain, dx = dy = 1 km
- Stretched vertical grid: dz = 50 m at the surface, <= 100 m below 3 km,
  growing to ~500 m aloft; model top at 33 km (105 levels)
- Fixed SST (295 / 300 / 305 K); interactive surface fluxes from the
  surface-layer (MOST) scheme with Charnock sea roughness and automatic
  saturated-surface humidity (`erf.is_land = 0`)
- SAM single-moment microphysics with ice
- RRTMGP radiation with the RCEMIP insolation (S0 = 551.58 W/m2 at a fixed
  zenith angle of 42.05 deg, i.e. 409.6 W/m2, no diurnal cycle), surface
  albedo 0.07, CO2 = 348 ppmv, CH4 = 1650 ppbv, N2O = 306 ppbv, and the
  RCEMIP analytic ozone profile
- Rayleigh damping of winds above 25 km
- Initial condition: the RCEMIP analytic sounding (Eqs. 2-6 of the paper)
  read via `erf.init_type = input_sounding`, plus the protocol's
  symmetry-breaking thermal noise in the five lowest model layers (amplitude
  0.1 K in the lowest layer decreasing linearly to 0.02 K in the fifth;
  `erf.prob_name = "RCEMIP"` dispatches to
  `Source/Prob/ERF_InitCustomPert_RCEMIP.H`)

The analytic sounding and ozone profile have been verified line-by-line
against A. Wing's reference implementation (`rcemip_on_z.m` /
`create_snd_analytic.m`, linked from https://myweb.fsu.edu/awing/rcemipsims.html)
and the constants in Tables 1-2 and Eq. 1 of the paper.

## Files

- `generate_rcemip_inputs.py` — generates the analytic soundings for all
  three SSTs, the stretched `erf.terrain_z_levels`, the `erf.o3vmr` profile
  on that grid's cell centers, and reference values (qsat(SST), cos(zenith)).
  Rerun it if you change the vertical grid, and paste the printed
  `terrain_z_levels`/`o3vmr`/`n_cell` lines into the inputs deck. `--plot`
  writes a figure of the initial profiles (requires matplotlib);
  `--official-grid` emits the exact 74-level grid of Table 3 of the paper
  (cell centers on the published levels, top face at 33250 m) instead of the
  finer default grid, for protocol-exact comparability.
- `ekat-apple-silicon-fpe.patch` — required to build EKAT (pulled in by
  RRTMGP) on Apple Silicon: EKAT's Apple fallback for `feenableexcept` was
  written for x86 `fenv_t` and does not compile on arm64. Apply with
  `git -C Submodules/ekat apply ../../Exec/CanonicalTests/RCEMIP/ekat-apple-silicon-fpe.patch`
  after any submodule update.
- `input_sounding_rcemip_sst295/300/305` — generated soundings
  (`z  theta[K]  qv[g/kg]  u  v`; first line is the surface reference).
- `inputs_rcemip_small` — the RCE_small deck at SST = 300 K.

## Building

The case uses the shared ERF executable with RRTMGP enabled. With GNU Make:

```sh
cd Exec
make -j USE_RRTMGP=TRUE USE_NETCDF=TRUE
```

The RRTMGP k-distribution and cloud-optics NetCDF files ship with the RRTMGP
submodule; the deck points at them with paths relative to this directory
(`erf.rrtmgp_file_path = ../../../Submodules/RRTMGP`). Adjust if you run from
elsewhere.

## Running the other SSTs

Change four lines (or override on the command line):

```
erf.input_sounding_file = "input_sounding_rcemip_sst305"
erf.most.surf_temp      = 305.0
erf.most.surf_moist     = 0.029441   # qsat(SST,p0): 295K->0.016209, 305K->0.029441
erf.rad_t_sfc           = 305.0
```

## Notes / deviations from the protocol

- The analytic sounding uses the RCEMIP constants (g = 9.79764 m/s2,
  Rd = 287.04) to evaluate T(z), p(z), q(z); potential temperature in the
  sounding file is then formed with ERF's own constants so ERF's EOS
  reproduces the intended temperature profile. ERF re-integrates the
  hydrostatic base state from (theta, qv) with its own gravity (9.81),
  so the initial pressure differs slightly from Eq. 5/6. RCE forgets its
  initial condition, so this only matters for the first hours.
- `erf.most.surf_temp` is used both as the surface theta in the MOST flux
  iteration and as the temperature in qsat at the surface; at
  p_sfc = 1014.8 hPa the theta/T distinction is ~1.3 K. The deck sets it to
  the SST.
- The protocol asks each model to use its own surface-flux scheme; here that
  is the Moeng MOST formulation with a constant roughness length and a
  prescribed saturated surface (`erf.most.surf_moist = qsat(SST, p0)`). ERF's
  sea-state roughness options (Charnock etc.) diverge at cold start with the
  protocol's zero-wind initial state (the z0/ustar fixed-point iteration in
  `ERF_MOSTStress.H` fails as ustar -> 0) — a candidate upstream fix; until
  then the deck routes the ocean through the constant-z0 (land) path, which
  with fixed SST is thermodynamically identical. The protocol's optional
  1 m/s minimum surface wind is not directly settable in ERF (the MOST
  iteration has a hardcoded 0.1 m/s floor); the deck instead enables
  `erf.most.include_wstar`, ERF's convective-gustiness term, which plays the
  same role in the weak-wind limit (the protocol also allows the quadrature
  form).
- `erf.rad_freq_in_steps = 10` calls radiation every 60 s; the protocol
  allows any interval <= 15 min, so this can be relaxed for speed.
- Timestep (`fixed_dt = 6 s`, `fixed_fast_dt = 1 s`) has not been tuned;
  reduce if the run goes unstable during deep-convection onset.
- `RCE_large` (channel, ~6000 km x 400 km, dx = 3 km) is not set up yet; it
  needs only a different deck (domain size, dx, and its own z levels /
  o3vmr from the generator script).

## Requested RCEMIP outputs

The protocol's 0D/1D/2D/3D output lists (Tables 6-9 of Wing et al. 2018) are
not wired up; plotfiles include the state and moisture fields once per
simulated day. ERF data logs / column diagnostics can be added to the deck as
needed.
